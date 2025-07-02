import ee
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math # For math.ceil

class GlobalWaterMaskingPipeline:
    def __init__(self, project_id=None):
        try:
            # Check if Earth Engine is already initialized to avoid re-initializing
            # This is a bit tricky; ee.data._initialized is not public.
            # A more robust check might involve a dummy GEE call.
            if not hasattr(ee.data, '_credentials') or ee.data._credentials is None:
                if project_id:
                    ee.Initialize(project=project_id)
                else:
                    ee.Initialize()
                print("Google Earth Engine initialized successfully")
            else:
                print("Google Earth Engine already initialized.")
                if project_id and ee.data._cloud_api_user_project != project_id:
                     print(f"Warning: GEE initialized with project '{ee.data._cloud_api_user_project}', but '{project_id}' was requested. Cannot change project after initialization.")

        except Exception as e:
            print(f"Error initializing GEE: {e}")
            print("Please authenticate with: ee.Authenticate()")

    def get_global_grid_generator(self, grid_size=10, max_lat=80, min_lat=-60):
        """Generator that yields grid tiles one at a time to avoid memory issues."""
        for lon in range(-180, 180, grid_size):
            # Adjust max_lat to be inclusive for the last tile if it aligns with max_lat
            current_max_lat = max_lat + grid_size - 1 if (max_lat % grid_size != 0 and max_lat > 0) else max_lat
            for lat in range(min_lat, current_max_lat, grid_size):
                bounds = ee.Geometry.Rectangle([
                    lon, lat,
                    min(lon + grid_size, 180),
                    min(lat + grid_size, max_lat) # Use original max_lat for clipping
                ])
                grid_id = f"{lon}_{lat}"
                feature = ee.Feature(bounds, {'grid_id': grid_id})
                yield {'feature': feature, 'grid_id': grid_id, 'geometry': bounds}

    def get_regional_grid(self, region_bounds, grid_size=10):
        """Get grid for a specific region to avoid global processing."""
        min_lon, min_lat, max_lon, max_lat = region_bounds
        grid = []
        # Calculate appropriate ranges to cover the whole extent
        start_lon = int(math.floor(min_lon / grid_size) * grid_size)
        start_lat = int(math.floor(min_lat / grid_size) * grid_size)
        end_lon = int(math.ceil(max_lon / grid_size) * grid_size)
        end_lat = int(math.ceil(max_lat / grid_size) * grid_size)

        for lon in range(start_lon, end_lon, grid_size):
            for lat in range(start_lat, end_lat, grid_size):
                bounds = ee.Geometry.Rectangle([
                    lon, lat,
                    min(lon + grid_size, max_lon), # Clip to region_bounds
                    min(lat + grid_size, max_lat)  # Clip to region_bounds
                ])
                # Ensure the created bounds intersect with the actual region_bounds
                # This is important for tiles that slightly extend beyond the given region_bounds
                actual_bounds = bounds.intersection(ee.Geometry.Rectangle(region_bounds), 1e-3) # Tolerance

                # Only add if the intersection is not empty
                if actual_bounds.area().getInfo() > 0:
                    grid_id = f"{lon}_{lat}"
                    feature = ee.Feature(actual_bounds, {'grid_id': grid_id})
                    grid.append({'feature': feature, 'grid_id': grid_id, 'geometry': actual_bounds})
        return grid

    def fetch_sentinel2_collection(self, start_date, end_date, cloud_cover_max=20, region=None):
        """Add region filtering to reduce collection size."""
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max)))
        
        if region:
            collection = collection.filterBounds(region)
            
        collection = collection.map(self._mask_s2_clouds)
        
        # Use getInfo() sparingly - only when necessary for monitoring
        try:
            size = collection.size().getInfo()
            print(f"Sentinel-2 collection size: {size}")
        except Exception as e:
            print(f"Sentinel-2 collection created (size check skipped due to error: {e})")
        
        return collection

    def _mask_s2_clouds(self, image):
        scl = image.select('SCL')
        # SCL values for clouds: 3 (cloud medium prob), 8 (cloud high prob), 9 (cirrus)
        # SCL values for shadow: 2 (cloud shadow)
        # We want to keep pixels that are NOT cloud/shadow.
        cloud_mask = scl.neq(2).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
        return image.updateMask(cloud_mask).copyProperties(image, ['system:time_start', 'system:index'])

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
        """Apply a mask based on custom band logic. Returns a single-band boolean mask."""
        if mask_expression is None:
            # If no custom mask, return an image where all pixels are True (unmasked)
            # This is important if you later expect a boolean image for consistency
            return ee.Image.constant(1).rename(['mask_true'])
        
        mask = mask_expression(image) # mask_expression is expected to return a boolean image
        return mask.rename(['custom_mask']) # Ensure consistent band name for later collection

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

    def export_to_asset(self, image, asset_id, description, region, scale=10, max_pixels=1e9, crs='EPSG:4326'):
        """Reduced max_pixels to prevent timeouts."""
        # CORRECTED CALL: Pass arguments as keyword arguments directly
        task = ee.batch.Export.image.toAsset(
            image=image, # This 'image' is your ee.Image object
            description=description,
            assetId=asset_id,
            region=region,
            scale=scale,
            maxPixels=max_pixels,
            crs=crs # Use EPSG:4326 as default for broader compatibility
        )
        task.start()
        print(f"Started export: {description}")
        return task

    def check_tile_has_water_efficient(self, tile_geom, jrc_water_mask, threshold=0.001):
        """Efficient water check using smaller scale and pixel limit."""
        try:
            # Use coarser scale for faster processing
            result = jrc_water_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=tile_geom,
                scale=300,  # Coarser scale for faster processing
                maxPixels=1e6,  # Lower pixel limit
                bestEffort=True,  # Allow approximation
                tileScale=4  # Use tiling for large areas
            )
            
            # Get the first band name
            band_name = jrc_water_mask.bandNames().get(0)
            water_fraction = result.get(band_name)
            
            # Convert to number and handle null values
            water_fraction = ee.Number(ee.Algorithms.If(water_fraction, water_fraction, 0))
            
            return water_fraction.gt(threshold)
            
        except Exception as e:
            print(f"Error checking water for tile: {e}")
            return ee.Number(0).gt(threshold)  # Default to False

    def process_single_tile(self, tile_dict, s2_collection_full, jrc_water, custom_mask_fn, 
                            output_asset_base, scale, water_threshold=0.001, years_for_consistency=None):
        """Process a single tile with error handling and consistency masking."""
        grid_id = tile_dict['grid_id']
        try:
            geom = tile_dict['geometry']
            
            # Check if tile has significant water coverage (server-side boolean)
            has_water_ee = self.check_tile_has_water_efficient(geom, jrc_water, water_threshold)
            
            # This function will be mapped over years for consistency.
            # It expects a single year's S2 collection.
            def generate_yearly_solar_panel_mask(year_str):
                year_start = ee.Date(f'{year_str}-01-01')
                year_end = ee.Date(f'{int(year_str) + 1}-01-01')
                
                s2_yearly_collection = s2_collection_full.filterDate(year_start, year_end).filterBounds(geom)
                
                # Check if there are images in the yearly collection for this tile
                tile_size = s2_yearly_collection.size()
                
                # Create composite for the year
                composite_yearly = ee.Algorithms.If(
                    tile_size.gt(0),
                    self.create_composite(s2_yearly_collection),
                    # Dummy image with expected bands if no S2 images for the year/tile
                    ee.Image.constant(0).rename(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                )
                composite_yearly = ee.Image(composite_yearly)

                # Apply custom solar panel mask (returns a single boolean band)
                solar_panel_mask = self.apply_custom_mask(composite_yearly, custom_mask_fn)

                # Combine with JRC water mask (JRC water mask is already pre-filtered by tile extent)
                # Ensure jrc_water is also clipped to the current tile geometry for precise overlay
                jrc_water_clipped = jrc_water.clip(geom) # Clip JRC for this tile
                combined_mask = solar_panel_mask.And(jrc_water_clipped.gt(0)) # gt(0) ensures it's a binary 0/1 mask
                
                # We rename to 'mask' for consistency in the image collection
                return combined_mask.rename('mask').set('year', year_str) # Set year property for filtering

            # --- Main logic for processing a tile ---
            
            # The final image to export: if has_water, compute the mask, else a zero image.
            # We need to perform the consistency mask calculation if has_water_ee is true.
            
            # Define the computation for the "has water" case
            def compute_mask_for_water_tile():
                if years_for_consistency and custom_mask_fn:
                    # Create an ImageCollection of yearly solar panel masks
                    yearly_masks_collection = ee.ImageCollection(
                        [generate_yearly_solar_panel_mask(year) for year in years_for_consistency]
                    )
                    
                    # Calculate the mean across the yearly masks (server-side equivalent of np.mean)
                    # This will give a float image where pixel value is the fraction of years it was masked
                    consistency_mean = yearly_masks_collection.mean()
                    
                    # Apply consistency threshold
                    # This creates the final binary mask (1 if consistent, 0 otherwise)
                    consistent_mask = consistency_mean.gte(0.1) # Threshold 0.1 (10% consistency)

                    # For export, we want a binary mask. We'll ensure it's a single band.
                    return consistent_mask.rename('consistent_solar_panels').clip(geom)
                else:
                    # If no years for consistency or no custom mask, export a dummy mask
                    # or perhaps a simple water mask (depending on your use case).
                    # For now, let's export a black image with a 'mask' band.
                    print(f"Warning: No years for consistency or custom_mask_fn provided for tile {grid_id}. Exporting a blank mask.")
                    return ee.Image.constant(0).rename('consistent_solar_panels').clip(geom)
            
            # Use ee.Algorithms.If to conditionally execute the mask computation
            final_image_to_export = ee.Algorithms.If(
                has_water_ee,
                compute_mask_for_water_tile(),
                # If no water, export an empty mask image (single band 'consistent_solar_panels')
                ee.Image.constant(0).rename('consistent_solar_panels').clip(geom)
            )
            
            final_image_to_export = ee.Image(final_image_to_export) # Cast needed after ee.Algorithms.If
            
            asset_id = f"{output_asset_base}_{grid_id}"
            desc = f"consistent_solar_panels_{grid_id}"
            
            # For exporting an entire region as one asset, you would call export_to_asset ONCE
            # outside the loop of process_single_tile. The 'process_single_tile' logic
            # is primarily for complex, tile-specific intermediate steps or if you
            # absolutely needed separate tile exports.
            # However, since the user asked to combine "the whole thing", we'll adjust the main
            # `run_batch_pipeline` to manage this.
            
            # For now, this function still returns a task per tile.
            # We'll adjust `run_batch_pipeline` to create one consolidated export task.
            
            # This function returns the GEE image object itself, not a task,
            # as the task will be created at a higher level (for the whole region).
            return final_image_to_export.set({'grid_id': grid_id}) # Attach grid_id property for debugging/consolidation
            
        except Exception as e:
            print(f"Error processing tile {grid_id}: {e}")
            # Return a dummy empty image if there's an error
            return ee.Image.constant(0).rename('consistent_solar_panels').clip(geom).set({'grid_id': grid_id, 'error': str(e)})

    def run_batch_pipeline(self, start_date, end_date, output_asset_base,
                            region_bounds=None, batch_size=50, **kwargs):
        """Process tiles in batches and then combine for a single region export."""
        
        water_class = kwargs.get('water_class', 'permanent')
        custom_mask_fn = kwargs.get('custom_mask_fn', None)
        grid_size = kwargs.get('grid_size', 10)
        scale = kwargs.get('scale', 10)
        water_threshold = kwargs.get('water_threshold', 0.001)
        
        # New parameter for consistency mask years
        years_for_consistency = kwargs.get('years_for_consistency', None)
        if not years_for_consistency:
            print("Warning: 'years_for_consistency' not provided. Consistency mask might not function as expected.")
        
        print("Setting up pipeline...")
        
        # Get region-specific data if bounds provided
        if region_bounds:
            region_geom = ee.Geometry.Rectangle(region_bounds)
            s2_full_collection = self.fetch_sentinel2_collection(start_date, end_date, region=region_geom)
            # Fetch tiles that cover the defined region_bounds, potentially clipping them.
            tiles = self.get_regional_grid(region_bounds, grid_size)
        else:
            print("Error: Global processing without region bounds is not supported for full export.")
            print("Please provide 'region_bounds' for the area you wish to export.")
            return [] # Exit if no specific region for full export

        jrc = self.fetch_jrc_water_dataset(water_class)
        
        print(f"Generating masks for {len(tiles)} tiles in batches of {batch_size}")
        
        # This will store ee.Image objects, not tasks
        all_processed_images_for_mosaic = []
        
        # Process tiles in batches, but only fetch the *GEE image objects*
        # The actual computations will only run when we export the final mosaic.
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i+batch_size]
            print(f"Preparing batch {i//batch_size + 1}/{(len(tiles)-1)//batch_size + 1}")
            
            for tile in batch:
                # process_single_tile now returns the ee.Image object
                processed_image = self.process_single_tile(
                    tile, s2_full_collection, jrc, custom_mask_fn, output_asset_base, 
                    scale, water_threshold, years_for_consistency
                )
                if processed_image:
                    all_processed_images_for_mosaic.append(processed_image)
            
            # Small delay between batches if needed, but for server-side GEE operations,
            # this often isn't strictly necessary for API limits, more for logging.
            time.sleep(1)

        if not all_processed_images_for_mosaic:
            print("No images were processed for mosaic. Exiting.")
            return []

        # Convert the list of ee.Image objects into an ee.ImageCollection
        image_collection_to_mosaic = ee.ImageCollection(all_processed_images_for_mosaic)
        
        print("Creating mosaic from processed tiles...")
        # Mosaic all the processed images into a single image.
        # Ensure that the images have the same band name for mosaicing.
        # Our `process_single_tile` now renames the output to 'consistent_solar_panels'.
        
        # Define how to resolve overlaps (e.g., 'mean', 'mosaic' which takes the last).
        # For a binary mask, 'max' or 'or' is generally suitable to ensure any detected pixel is kept.
        # If the input images are already masked (0 for no data), `mosaic()` effectively merges them.
        final_mosaic = image_collection_to_mosaic.mosaic() 
        
        # Now, export this single, large mosaic to an asset.
        # The output_asset_base should refer to the name of the final combined asset.
        # We need to ensure the max_pixels is very large for the entire region.
        
        final_asset_id = f"{output_asset_base}_combined" # Appending _combined to indicate it's one asset
        final_description = f"Consistent_Solar_Panels_Mask_{region_bounds[0]}_{region_bounds[1]}_to_{region_bounds[2]}_{region_bounds[3]}"
        
        print(f"Starting export of final combined mosaic to asset: {final_asset_id}")
        export_task = self.export_to_asset(
            final_mosaic, 
            final_asset_id, 
            final_description, 
            region_geom, # Use the overall region geometry for the export
            scale=scale, 
            max_pixels=1e13, # Set a very large max_pixels for large regional exports
            crs='EPSG:4326' # Ensure CRS consistency
        )
        
        return [export_task] # Return a list containing the single export task

    def run_regional_pipeline(self, start_date, end_date, output_asset_base,
                              region_name, region_bounds, **kwargs):
        """Process a specific region and export as a single combined asset."""
        print(f"Processing region: {region_name}")
        print(f"Bounds: {region_bounds}")
        
        # Here, `output_asset_base` should already include the user's base path
        # and we append the region name later for the final asset.
        return self.run_batch_pipeline(
            start_date, end_date, f"{output_asset_base}_{region_name}", # Base for the final asset
            region_bounds=region_bounds, **kwargs
        )

    def monitor_tasks(self, tasks, interval=60, max_wait_hours=24):
        """Monitor tasks with timeout to avoid infinite loops."""
        print("Monitoring tasks...")
        start_time = time.time()
        max_wait_seconds = max_wait_hours * 3600
        
        # Ensure tasks is iterable even if empty
        if not tasks:
            print("No tasks to monitor.")
            return {'COMPLETED': 0, 'FAILED': 0, 'RUNNING': 0, 'READY': 0, 'CANCEL_REQUESTED': 0, 'TOTAL': 0}

        # Use a list comprehension to ensure all elements are actual ee.batch.Task objects
        valid_tasks = [task for task in tasks if isinstance(task, ee.batch.Task)]
        if not valid_tasks:
            print("No valid GEE tasks to monitor.")
            return {'COMPLETED': 0, 'FAILED': 0, 'RUNNING': 0, 'READY': 0, 'CANCEL_REQUESTED': 0, 'TOTAL': 0}
            
        print(f"Monitoring {len(valid_tasks)} GEE tasks.")

        while True:
            current_time = time.time()
            if current_time - start_time > max_wait_seconds:
                print(f"Stopping monitoring after {max_wait_hours} hours")
                break
                
            states = {'COMPLETED': 0, 'FAILED': 0, 'RUNNING': 0, 'READY': 0, 'CANCEL_REQUESTED': 0}
            
            for task in valid_tasks:
                try:
                    status = task.status() # Fetch status once per task per iteration
                    state = status['state']
                    states[state] = states.get(state, 0) + 1
                    if state == 'FAILED':
                        print(f"FAILED Task {task.id}: {status['error_message']}") # Print error message
                except Exception as e:
                    print(f"Error checking task {getattr(task, 'id', 'unknown')}: {e}")
                    states['ERROR'] = states.get('ERROR', 0) + 1
            
            total_tasks = len(valid_tasks)
            completed_or_failed = states['COMPLETED'] + states['FAILED']
            
            print(f"Task states: {states} (Total: {total_tasks}, Done: {completed_or_failed})")
            
            if completed_or_failed == total_tasks:
                print("All tasks completed or failed!")
                break
                
            time.sleep(interval)
        
        return states

# EXAMPLE USAGE WITH REGIONAL PROCESSING
if __name__ == '__main__':
    # Ensure you have authenticated GEE locally: ee.Authenticate()
    # If running in Colab, this is usually handled.
    # ee.Authenticate() # Uncomment if not already authenticated

    # Replace with your actual Google Cloud Project ID
    pipeline = GlobalWaterMaskingPipeline(project_id='summer-job-ife') 
    
    def solar_panel_mask_expression(image):
        """
        Returns a binary mask for likely water-based solar panels based on Sentinel-2 bands.
        This function operates on an ee.Image object.
        """
        # Get bands
        blue = image.select('B2')
        green = image.select('B3')
        red = image.select('B4')
        nir = image.select('B8')
        swir_1 = image.select('B11')
        swir_2 = image.select('B12')

        # Spectral index conditions (ensure 1e-8 for division by zero safety)
        bi_green = (blue.subtract(green)).divide(blue.add(green).add(1e-8)).gt(-0.07)
        bi_red = (blue.subtract(red)).divide(blue.add(red).add(1e-8)).gt(-0.05)
        ndbi = (swir_1.subtract(nir)).divide(swir_1.add(nir).add(1e-8)).gt(0.02)
        nsdsi = (swir_1.subtract(swir_2)).divide(swir_1.add(swir_2).add(1e-8)).gt(0.12)
        swir_thresh = swir_1.gt(0.07)

        # Combine all conditions
        mask = bi_green.And(bi_red).And(ndbi).And(nsdsi).And(swir_thresh)
        return mask.rename('solar_panel_potential') # Rename for clarity
    
    # Define regions to process. For Europe, you could use a more precise geometry.
    regions = {
        'europe_test_area': [5.0, 58.0, 10.0, 62.0], # A smaller test area in Norway/Sweden
        # 'europe': [-10, 35, 40, 70], # Full Europe (will take a very long time to export)
    }
    
    # Define the years for consistency masking
    # These years will be used to filter the S2 collection for generating yearly masks.
    # It's crucial that these years have data in your S2 collection.
    years_for_consistency_analysis = [str(y) for y in range(2021, 2024)] # E.g., 2021, 2022, 2023
    
    # Base asset path for your exports
    base_output_asset_path = 'users/sigurdvargdal/consistent_solar_panels' 
    
    # Process each region separately
    for region_name, bounds in regions.items():
        print(f"\n=== Processing {region_name} ===")
        
        tasks = pipeline.run_regional_pipeline(
            start_date=f'{years_for_consistency_analysis[0]}-01-01', # Start from the first year
            end_date=f'{int(years_for_consistency_analysis[-1]) + 1}-01-01', # End after the last year
            output_asset_base=base_output_asset_path,
            region_name=region_name,
            region_bounds=bounds,
            water_class='permanent',
            custom_mask_fn=solar_panel_mask_expression,
            grid_size=5, # Keep grid_size relatively small for better parallelization within GEE
            scale=10,    # Output resolution in meters
            batch_size=25, # Number of tiles to *prepare* at once (GEE handles server-side parallelism)
            water_threshold=0.001,
            years_for_consistency=years_for_consistency_analysis # Pass the years list
        )
        
        if tasks:
            # Monitor this region's tasks (should be just one export task for the combined asset)
            final_states = pipeline.monitor_tasks(tasks, interval=300, max_wait_hours=24) # Increased max_wait_hours
            print(f"Region {region_name} completed with states: {final_states}")
            
            # Print asset ID for easy access in GEE Code Editor
            if final_states.get('COMPLETED', 0) > 0:
                print(f"Final combined asset for {region_name} is: {tasks[0].config['assetId']}")

        # Wait between regions to avoid API limits if processing multiple large regions
        time.sleep(60)