import ee
import geemap
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import os
from pathlib import Path
import json

class GlobalWaterSegmentationPipeline:
    """
    Pipeline for global water segmentation using JRC water data,
    followed by ML-based false positive removal
    """
    
    def __init__(self, grid_size_degrees=5, output_dir='water_segments'):
        """
        Initialize the pipeline
        
        Args:
            grid_size_degrees: Size of processing grid in degrees
            output_dir: Directory for output files
        """
        ee.Initialize()
        self.grid_size = grid_size_degrees
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # JRC Global Surface Water dataset
        self.jrc_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        
    def create_global_grid(self):
        """Create a global grid for processing"""
        grid_cells = []
        
        # Global bounds
        for lat in range(-90, 90, self.grid_size):
            for lon in range(-180, 180, self.grid_size):
                bounds = [
                    lon, lat, 
                    min(lon + self.grid_size, 180), 
                    min(lat + self.grid_size, 90)
                ]
                grid_cells.append({
                    'bounds': bounds,
                    'id': f'grid_{lat}_{lon}'
                })
        
        return grid_cells
    
    def create_water_mask(self, bounds, water_threshold=50):
        """
        Create water mask for a specific region using JRC data
        
        Args:
            bounds: [west, south, east, north] in degrees
            water_threshold: Water occurrence threshold (0-100)
        """
        region = ee.Geometry.Rectangle(bounds)
        
        # Use water occurrence layer
        water_occurrence = self.jrc_water.select('occurrence')
        
        # Create mask where water occurrence > threshold
        water_mask = water_occurrence.gt(water_threshold)
        
        # Optional: Add seasonal water
        seasonal_water = self.jrc_water.select('seasonality').gt(6)  # Present >6 months
        combined_mask = water_mask.Or(seasonal_water)
        
        # Remove transitions (areas that changed significantly)
        transitions = self.jrc_water.select('transition')
        stable_water = combined_mask.And(transitions.eq(0))
        
        return stable_water.clip(region)
    
    def extract_water_polygons(self, water_mask, region, min_area_m2=1000):
        """
        Extract water polygons from mask and convert to shapefile format
        
        Args:
            water_mask: EE Image with water mask
            region: EE Geometry for the region
            min_area_m2: Minimum area threshold in square meters
        """
        # Convert to vectors
        vectors = water_mask.reduceToVectors(
            geometry=region,
            scale=30,  # 30m resolution
            geometryType='polygon',
            eightConnected=False,
            maxPixels=1e9
        )
        
        # Filter by area
        vectors_filtered = vectors.filter(ee.Filter.gte('count', min_area_m2 / 900))  # ~30m pixels
        
        return vectors_filtered
    
    def process_grid_cell(self, grid_cell):
        """Process a single grid cell"""
        bounds = grid_cell['bounds']
        cell_id = grid_cell['id']
        
        print(f"Processing grid cell: {cell_id}")
        
        # Skip if mostly ocean
        if self.is_mostly_ocean(bounds):
            return None
            
        try:
            # Create water mask
            water_mask = self.create_water_mask(bounds)
            region = ee.Geometry.Rectangle(bounds)
            
            # Extract polygons
            water_polygons = self.extract_water_polygons(water_mask, region)
            
            # Export to GeoDataFrame
            gdf = geemap.ee_to_gdf(water_polygons)
            
            if len(gdf) > 0:
                # Save as shapefile
                output_path = self.output_dir / f"{cell_id}_water.shp"
                gdf.to_file(output_path)
                return output_path
            
        except Exception as e:
            print(f"Error processing {cell_id}: {str(e)}")
            return None
    
    def is_mostly_ocean(self, bounds, ocean_threshold=0.8):
        """Check if grid cell is mostly ocean"""
        region = ee.Geometry.Rectangle(bounds)
        
        # Use SRTM or other land mask
        elevation = ee.Image("USGS/SRTMGL1_003")
        land_mask = elevation.gt(0)
        
        # Calculate land percentage
        land_stats = land_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            maxPixels=1e9
        )
        
        land_fraction = land_stats.getInfo().get('elevation', 0)
        return land_fraction < (1 - ocean_threshold)
    
    def merge_shapefiles(self):
        """Merge all generated shapefiles into one global shapefile"""
        shapefiles = list(self.output_dir.glob("*_water.shp"))
        
        if not shapefiles:
            print("No shapefiles found to merge")
            return None
            
        # Read and combine all shapefiles
        gdfs = []
        for shp_path in shapefiles:
            try:
                gdf = gpd.read_file(shp_path)
                gdf['source_grid'] = shp_path.stem
                gdfs.append(gdf)
            except Exception as e:
                print(f"Error reading {shp_path}: {e}")
        
        if gdfs:
            combined_gdf = gpd.pd.concat(gdfs, ignore_index=True)
            
            # Remove duplicates and clean geometry
            combined_gdf = combined_gdf.drop_duplicates()
            combined_gdf = combined_gdf[combined_gdf.geometry.is_valid]
            
            # Save global shapefile
            global_output = self.output_dir / "global_water_segments.shp"
            combined_gdf.to_file(global_output)
            
            print(f"Global shapefile saved: {global_output}")
            return global_output
    
    def upload_to_gee_as_asset(self, shapefile_path, asset_name):
        """
        Upload shapefile to Google Earth Engine as an asset
        Note: This requires using the Earth Engine command line tool
        """
        upload_command = f"""
        earthengine upload table --asset_id=users/YOUR_USERNAME/{asset_name} {shapefile_path}
        """
        
        print("To upload to GEE, run this command:")
        print(upload_command)
        print("\nOr use the GEE web interface to upload the shapefile manually")