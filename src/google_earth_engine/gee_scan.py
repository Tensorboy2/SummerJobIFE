import ee
import time

class GlobalWaterMaskingPipeline:
    def __init__(self, project_id=None):
        try:
            if project_id:
                ee.Initialize(project=project_id)
            else:
                ee.Initialize()
            print("Google Earth Engine initialized successfully")
        except Exception as e:
            print(f"Error initializing GEE: {e}")
            print("Please authenticate with: ee.Authenticate()")

    def get_global_grid(self, grid_size=10):
        global_bounds = ee.Geometry.Rectangle([-180, -60, 180, 80])
        grid = []
        for lon in range(-180, 180, grid_size):
            for lat in range(-60, 80, grid_size):
                bounds = ee.Geometry.Rectangle([
                    lon, lat,
                    min(lon + grid_size, 180),
                    min(lat + grid_size, 80)
                ])
                grid.append(ee.Feature(bounds, {'grid_id': f"{lon}_{lat}"}))
        return ee.FeatureCollection(grid)

    def fetch_sentinel2_collection(self, start_date, end_date, cloud_cover_max=20):
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
                     .map(self._mask_s2_clouds))
        print(f"Sentinel-2 collection size: {collection.size().getInfo()}")
        return collection

    def _mask_s2_clouds(self, image):
        scl = image.select('SCL')
        cloud_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
        return image.updateMask(cloud_mask).copyProperties(image, ['system:time_start'])

    def fetch_jrc_water_dataset(self, water_class='permanent'):
        dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
        if water_class == 'permanent':
            return dataset.select('occurrence').gt(75).rename(['permanent_water'])
        elif water_class == 'occurrence':
            return dataset.select('occurrence')
        elif water_class == 'seasonal':
            occ = dataset.select('occurrence')
            sea = dataset.select('seasonality')
            return (occ.gt(25).And(occ.lt(75)).Or(sea.gt(5))).rename(['seasonal_water'])
        elif water_class == 'transitions':
            return dataset.select('transition')
        else:
            raise ValueError("Invalid water_class")

    def apply_custom_mask(self, image, mask_expression):
        """Apply a mask based on custom band logic."""
        mask = mask_expression(image)
        return image.updateMask(mask)

    def create_composite(self, collection, method='median'):
        if method == 'median':
            return collection.median()
        elif method == 'mean':
            return collection.mean()
        elif method == 'max':
            return collection.max()
        elif method == 'min':
            return collection.min()
        else:
            raise ValueError("Invalid composite method")

    def export_to_asset(self, image, asset_id, description, region, scale=10, max_pixels=1e13):
        task = ee.batch.Export.image.toAsset({
            'image': image,
            'description': description,
            'assetId': asset_id,
            'region': region,
            'scale': scale,
            'maxPixels': max_pixels,
            'crs': 'EPSG:4326'
        })
        task.start()
        print(f"Started export: {description}")
        return task

    def filter_grid_by_water(self, grid_fc, jrc_water_mask, threshold=0.001):
        def tile_has_water(feature):
            geom = feature.geometry()
            frac_dict = jrc_water_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=30,
                maxPixels=1e9
            )
            frac = ee.Number(frac_dict.get(jrc_water_mask.bandNames().get(0)))  # safer access
            
            # If frac is null, replace with 0 to avoid error
            frac_safe = ee.Algorithms.If(frac, frac, 0)
            
            return ee.Algorithms.If(
                ee.Number(frac_safe).gt(threshold),
                feature,
                None
            )

        return grid_fc.map(tile_has_water).filter(ee.Filter.notNull(['grid_id']))

    def run_tile_based_pipeline(self, start_date, end_date, output_asset_base,
                                water_class='permanent', custom_mask_fn=None,
                                grid_size=10, scale=10):
        print("Fetching data and setting up pipeline...")
        s2 = self.fetch_sentinel2_collection(start_date, end_date)
        jrc = self.fetch_jrc_water_dataset(water_class)
        grid = self.get_global_grid(grid_size)
        tiles = self.filter_grid_by_water(grid, jrc)
        tasks = []

        tile_list = tiles.toList(tiles.size())
        for i in range(tiles.size().getInfo()):
            tile = ee.Feature(tile_list.get(i))
            geom = tile.geometry()
            grid_id = tile.get('grid_id').getInfo()
            try:
                s2_tile = s2.filterBounds(geom)
                composite = self.create_composite(s2_tile)
                masked = self.apply_custom_mask(composite, custom_mask_fn)
                asset_id = f"{output_asset_base}_{grid_id}"
                desc = f"s2_masked_{grid_id}"
                task = self.export_to_asset(masked, asset_id, desc, geom, scale)
                tasks.append(task)
            except Exception as e:
                print(f"Error processing tile {grid_id}: {e}")

        return tasks

    def monitor_tasks(self, tasks, interval=60):
        print("Monitoring tasks...")
        while True:
            states = {'COMPLETED': 0, 'FAILED': 0, 'RUNNING': 0, 'READY': 0}
            for task in tasks:
                state = task.status()['state']
                states[state] += 1
                if state == 'FAILED':
                    print(f"FAILED: {task.status()}")
            print(states)
            if states['COMPLETED'] + states['FAILED'] == len(tasks):
                break
            time.sleep(interval)

# EXAMPLE USAGE
if __name__ == '__main__':
    pipeline = GlobalWaterMaskingPipeline(project_id='summer-job-ife')
    def solar_panel_mask_expression(image):
        """
        Returns a binary mask for likely water-based solar panels based on Sentinel-2 bands.
        
        Earth Engine bands:
            B2  = Blue
            B3  = Green
            B4  = Red
            B8  = NIR
            B11 = SWIR1
            B12 = SWIR2
        """
        # Get bands
        blue = image.select('B2')
        green = image.select('B3')
        red = image.select('B4')
        nir = image.select('B8')
        swir_1 = image.select('B11')
        swir_2 = image.select('B12')

        # Spectral index conditions
        bi_green = (blue.subtract(green)).divide(blue.add(green).add(1e-8)).gt(-0.07)
        bi_red = (blue.subtract(red)).divide(blue.add(red).add(1e-8)).gt(-0.05)
        ndbi = (swir_1.subtract(nir)).divide(swir_1.add(nir).add(1e-8)).gt(0.02)
        nsdsi = (swir_1.subtract(swir_2)).divide(swir_1.add(swir_2).add(1e-8)).gt(0.12)
        swir_thresh = swir_1.gt(0.07)

        # Combine all conditions
        mask = bi_green.And(bi_red).And(ndbi).And(nsdsi).And(swir_thresh)
        return mask

    tasks = pipeline.run_tile_based_pipeline(
        start_date='2023-01-01',
        end_date='2023-12-31',
        output_asset_base='users/sigurdvargdal/s2_ndwi_masked',
        water_class='permanent',
        custom_mask_fn=solar_panel_mask_expression,
        grid_size=5,
        scale=10
    )
    pipeline.monitor_tasks(tasks)
