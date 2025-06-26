import ee
import numpy as np
from datetime import datetime, timedelta
import time

class GlobalWaterMaskingPipeline:
    """
    Pipeline for processing Sentinel-2 imagery with JRC water dataset masking
    and exporting results as Google Earth Engine assets.
    """
    
    def __init__(self, project_id=None):
        """
        Initialize the pipeline with Google Earth Engine authentication.
        
        Args:
            project_id (str): GEE project ID for authentication
        """
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
        """
        Create a global grid for processing large areas efficiently.
        
        Args:
            grid_size (int): Size of each grid cell in degrees
            
        Returns:
            ee.FeatureCollection: Global grid geometry
        """
        # Create global bounding box
        global_bounds = ee.Geometry.Rectangle([-180, -60, 180, 80])
        
        # Create grid
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
        """
        Fetch Sentinel-2 Level-2A surface reflectance collection.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            cloud_cover_max (int): Maximum cloud cover percentage
            
        Returns:
            ee.ImageCollection: Filtered Sentinel-2 collection
        """
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
                     .map(self._mask_s2_clouds))
        
        print(f"Sentinel-2 collection size: {collection.size().getInfo()}")
        return collection
    
    def _mask_s2_clouds(self, image):
        """
        Mask clouds and cloud shadows in Sentinel-2 imagery using SCL band.
        
        Args:
            image (ee.Image): Sentinel-2 image
            
        Returns:
            ee.Image: Cloud-masked image
        """
        scl = image.select('SCL')
        
        # SCL classification values:
        # 1: Saturated/Defective, 3: Cloud shadows, 8: Cloud medium probability
        # 9: Cloud high probability, 10: Thin cirrus, 11: Snow/ice
        cloud_shadow = scl.eq(3)
        cloud_medium = scl.eq(8)
        cloud_high = scl.eq(9)
        cirrus = scl.eq(10)
        snow = scl.eq(11)
        saturated = scl.eq(1)
        
        # Create mask (0 = masked, 1 = valid)
        mask = (cloud_shadow.Or(cloud_medium)
                .Or(cloud_high)
                .Or(cirrus)
                .Or(snow)
                .Or(saturated)
                .Not())
        
        return image.updateMask(mask).copyProperties(image, ['system:time_start'])
    
    def fetch_jrc_water_dataset(self, water_class='permanent'):
        """
        Fetch JRC Global Surface Water dataset.
        
        Args:
            water_class (str): Type of water classification
                - 'occurrence': Water occurrence frequency
                - 'permanent': Permanent water bodies
                - 'seasonal': Seasonal water bodies
                - 'transitions': Water transitions
                
        Returns:
            ee.Image: JRC water dataset
        """
        if water_class == 'occurrence':
            dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            return dataset.select('occurrence')
        
        elif water_class == 'permanent':
            dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            occurrence = dataset.select('occurrence')
            # Consider water permanent if occurrence > 75%
            return occurrence.gt(75).rename(['permanent_water'])
        
        elif water_class == 'seasonal':
            dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            occurrence = dataset.select('occurrence')
            seasonality = dataset.select('seasonality')
            # Seasonal water: occurrence 25-75% or seasonality indicates seasonal
            return (occurrence.gt(25).And(occurrence.lt(75))
                   .Or(seasonality.gt(5))).rename(['seasonal_water'])
        
        elif water_class == 'transitions':
            dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            return dataset.select('transition')
        
        else:
            raise ValueError("Invalid water_class. Use: 'occurrence', 'permanent', 'seasonal', or 'transitions'")
    
    def apply_water_masking_logic(self, sentinel2_image, jrc_water_mask, mask_type='exclude_water'):
        """
        Apply custom water masking logic to Sentinel-2 imagery.
        
        Args:
            sentinel2_image (ee.Image): Sentinel-2 image
            jrc_water_mask (ee.Image): JRC water mask
            mask_type (str): Type of masking logic
                - 'exclude_water': Mask out water pixels
                - 'water_only': Keep only water pixels
                - 'enhance_water': Enhance water detection
                
        Returns:
            ee.Image: Masked Sentinel-2 image
        """
        if mask_type == 'exclude_water':
            # Mask out water pixels (set to 0 where water exists)
            water_mask = jrc_water_mask.eq(0)  # Invert mask
            return sentinel2_image.updateMask(water_mask)
        
        elif mask_type == 'water_only':
            # Keep only water pixels
            return sentinel2_image.updateMask(jrc_water_mask)
        
        elif mask_type == 'enhance_water':
            # Enhanced water detection using NDWI and JRC data
            ndwi = self._calculate_ndwi(sentinel2_image)
            
            # Combine NDWI threshold with JRC water occurrence
            enhanced_water = ndwi.gt(0.1).Or(jrc_water_mask.gt(0))
            
            # Add enhanced water as a band
            return sentinel2_image.addBands(
                enhanced_water.rename(['enhanced_water']).byte()
            )
        
        else:
            raise ValueError("Invalid mask_type. Use: 'exclude_water', 'water_only', or 'enhance_water'")
    
    def _calculate_ndwi(self, image):
        """
        Calculate Normalized Difference Water Index (NDWI).
        
        Args:
            image (ee.Image): Sentinel-2 image
            
        Returns:
            ee.Image: NDWI image
        """
        return image.normalizedDifference(['B3', 'B8']).rename(['NDWI'])
    
    def create_composite(self, collection, composite_method='median'):
        """
        Create composite from image collection.
        
        Args:
            collection (ee.ImageCollection): Input collection
            composite_method (str): Compositing method ('median', 'mean', 'max', 'min')
            
        Returns:
            ee.Image: Composite image
        """
        if composite_method == 'median':
            return collection.median()
        elif composite_method == 'mean':
            return collection.mean()
        elif composite_method == 'max':
            return collection.max()
        elif composite_method == 'min':
            return collection.min()
        else:
            raise ValueError("Invalid composite_method. Use: 'median', 'mean', 'max', or 'min'")
    
    def export_to_asset(self, image, asset_id, description, region=None, scale=10, max_pixels=1e13):
        """
        Export processed image to Google Earth Engine asset.
        
        Args:
            image (ee.Image): Image to export
            asset_id (str): Asset ID for the exported image
            description (str): Description for the export task
            region (ee.Geometry): Region to export (None for global)
            scale (int): Export resolution in meters
            max_pixels (float): Maximum number of pixels to export
            
        Returns:
            ee.batch.Task: Export task
        """
        if region is None:
            # Global export region (excluding extreme polar regions)
            region = ee.Geometry.Rectangle([-180, -60, 180, 80])
        
        task = ee.batch.Export.image.toAsset(**{
            'image': image,
            'description': description,
            'assetId': asset_id,
            'region': region,
            'scale': scale,
            'maxPixels': max_pixels,
            'crs': 'EPSG:4326'
        })
        
        task.start()
        print(f"Export task started: {description}")
        print(f"Asset ID: {asset_id}")
        return task
    
    def run_global_pipeline(self, start_date, end_date, output_asset_base, 
                           water_class='permanent', mask_type='exclude_water',
                           grid_size=10, scale=10):
        """
        Run the complete global processing pipeline.
        
        Args:
            start_date (str): Start date for Sentinel-2 data
            end_date (str): End date for Sentinel-2 data
            output_asset_base (str): Base path for output assets
            water_class (str): JRC water classification type
            mask_type (str): Water masking logic type
            grid_size (int): Grid size for tiling (degrees)
            scale (int): Output resolution (meters)
            
        Returns:
            list: List of export tasks
        """
        print("Starting global water masking pipeline...")
        
        # 1. Fetch datasets
        print("Fetching Sentinel-2 collection...")
        s2_collection = self.fetch_sentinel2_collection(start_date, end_date)
        
        print("Fetching JRC water dataset...")
        jrc_water = self.fetch_jrc_water_dataset(water_class)
        
        # 2. Create composite
        print("Creating Sentinel-2 composite...")
        s2_composite = self.create_composite(s2_collection, 'median')
        
        # 3. Apply water masking
        print("Applying water masking logic...")
        masked_composite = self.apply_water_masking_logic(
            s2_composite, jrc_water, mask_type
        )
        
        # 4. Export global asset
        global_asset_id = f"{output_asset_base}_global_{water_class}_{mask_type}"
        
        task = self.export_to_asset(
            image=masked_composite,
            asset_id=global_asset_id,
            description=f"Global_S2_JRC_Water_{water_class}_{mask_type}",
            scale=scale,
            max_pixels=1e13
        )
        
        return [task]
    
    def monitor_tasks(self, tasks, check_interval=60):
        """
        Monitor export task progress.
        
        Args:
            tasks (list): List of export tasks
            check_interval (int): Check interval in seconds
        """
        print("Monitoring export tasks...")
        
        while True:
            completed = 0
            failed = 0
            running = 0
            
            for task in tasks:
                status = task.status()['state']
                if status == 'COMPLETED':
                    completed += 1
                elif status == 'FAILED':
                    failed += 1
                    print(f"Task failed: {task.status()}")
                elif status in ['RUNNING', 'READY']:
                    running += 1
            
            print(f"Tasks - Completed: {completed}, Running: {running}, Failed: {failed}")
            
            if completed + failed == len(tasks):
                break
            
            time.sleep(check_interval)
        
        print("All tasks completed!")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GlobalWaterMaskingPipeline(project_id='summer-job-ife')
    
    # Define parameters
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    output_asset_base = 'users/sigurdvargdal/s2_jrc_water_masked'
    
    # Run pipeline
    tasks = pipeline.run_global_pipeline(
        start_date=start_date,
        end_date=end_date,
        output_asset_base=output_asset_base,
        water_class='permanent',
        mask_type='exclude_water',
        scale=10
    )
    
    # Monitor progress
    pipeline.monitor_tasks(tasks)
    
    print("Pipeline completed successfully!")
    print(f"Results saved to: {output_asset_base}_global_permanent_exclude_water")