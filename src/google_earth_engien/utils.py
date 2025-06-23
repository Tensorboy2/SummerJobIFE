"""
utils.py

Functions used in Google Earth Engine pipeline for fetching and processing satellite data.
"""
import ee
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tifffile
import torch
import json
import os
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for satellite data processing."""
    project_id: str = 'summer-job-ife'
    buffer_degrees: float = 0.012
    image_dimensions: str = '256x256'
    cloud_threshold: float = 35.0
    water_occurrence_threshold: float = 50.0
    crs: str = 'EPSG:4326'
    
    # Band indices for Sentinel-2
    bands: Dict[str, int] = None
    
    def __post_init__(self):
        if self.bands is None:
            self.bands = {
                'blue': 1, 'green': 2, 'red': 3, 'nir': 7, 
                'swir_1': 10, 'swir_2': 11
            }


class EarthEngineManager:
    """Manages Google Earth Engine initialization and authentication."""
    
    @staticmethod
    def initialize(project_id: str) -> bool:
        """Initialize Google Earth Engine with error handling."""
        try:
            ee.Initialize(project=project_id)
            print("Earth Engine initialized successfully.")
            return True
        except ee.EEException as e:
            print(f"Earth Engine initialization failed: {e}")
            print("Please run 'ee.Authenticate()' in your terminal/notebook if you haven't.")
            return False


class SentinelProcessor:
    """Handles Sentinel-2 data processing and downloading."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @staticmethod
    def mask_s2_clouds(image):
        """Masks clouds in Sentinel-2 image using the QA60 band."""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask).divide(10000)  # Convert to reflectance (0-1)

    def create_region(self, lon: float, lat: float) -> ee.Geometry:
        """Create bounding box region around coordinates."""
        buffer = self.config.buffer_degrees
        return ee.Geometry.BBox(
            lon - buffer, lat - buffer, 
            lon + buffer, lat + buffer
        )

    def get_annual_composite(self, lon: float, lat: float, year: int) -> Optional[ee.Image]:
        """Get annual median composite for a specific location and year."""
        region = self.create_region(lon, lat)
        
        dataset = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(f'{year}-01-01', f'{year}-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.config.cloud_threshold))
            .map(self.mask_s2_clouds)
        )

        count = dataset.size().getInfo()
        if count == 0:
            print(f"    Warning: No clear-sky images found for {lat:.4f}_{lon:.4f} in year {year}")
            return None

        try:
            composite = dataset.median()
            _ = composite.getInfo()  # Test if composite is valid
            return composite
        except Exception as e:
            print(f"    Warning: Failed to create composite for {lat:.4f}_{lon:.4f} in {year}: {e}")
            return None

    def download_rgb_png(self, composite: ee.Image, region: ee.Geometry, 
                        output_path: str) -> bool:
        """Download RGB PNG image."""
        vis_params = {
            'min': 0.0,
            'max': 0.3,
            'bands': ['B4', 'B3', 'B2'],
        }
        
        thumb_url = composite.getThumbURL({
            'dimensions': int(self.config.image_dimensions.split('x')[0]),
            'region': region,
            'format': 'png',
            **vis_params
        })

        try:
            response = requests.get(thumb_url)
            response.raise_for_status()
            
            img_pil = Image.open(BytesIO(response.content))
            img_pil.save(output_path)
            return True
        except Exception as e:
            print(f"    Failed to download RGB PNG: {e}")
            return False

    def download_multispectral_tif(self, composite: ee.Image, region: ee.Geometry, 
                                  output_path: str) -> bool:
        """Download multispectral GeoTIFF image."""
        bands_to_download = [
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 
            'B8A', 'B9', 'B11', 'B12'
        ]
        
        image_selected = composite.select(bands_to_download)
        
        download_params = {
            'crs': self.config.crs,
            'region': region.getInfo()['coordinates'],
            'format': 'GEO_TIFF',
            'dimensions': self.config.image_dimensions
        }

        try:
            download_url = image_selected.getDownloadURL(download_params)
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"    Failed to download multispectral TIFF: {e}")
            return False

    def download_water_mask(self, region: ee.Geometry, output_path: str) -> bool:
        """Download water occurrence mask."""
        try:
            water_dataset = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
            water_clip = water_dataset.clip(region)
            water_mask = water_clip.gt(self.config.water_occurrence_threshold)

            water_mask_url = water_mask.getDownloadURL({
                'crs': self.config.crs,
                'region': region.getInfo()['coordinates'],
                'format': 'GEO_TIFF',
                'dimensions': self.config.image_dimensions,
            })

            response = requests.get(water_mask_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"    Failed to download water mask: {e}")
            return False

    def download_torch_tensor(self, composite: ee.Image, region: ee.Geometry, 
                             output_path: str, lat: float, lon: float, year: int) -> bool:
        """Download image as PyTorch tensor."""
        # First download as temporary TIFF
        temp_path = output_path.replace('.pt', '_temp.tif')
        
        if not self.download_multispectral_tif(composite, region, temp_path):
            return False

        try:
            img_np = tifffile.imread(temp_path)
            
            # Ensure correct dimension order (channels first)
            if img_np.ndim == 3 and img_np.shape[0] < img_np.shape[-1]:
                img_np = np.moveaxis(img_np, -1, 0)

            img_tensor = torch.tensor(img_np, dtype=torch.float32)
            torch.save({
                'image': img_tensor, 
                'coords': (lat, lon), 
                'year': year
            }, output_path)
            
            os.remove(temp_path)  # Clean up temporary file
            return True
        except Exception as e:
            print(f"    Failed to create torch tensor: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False


class SolarPanelMask:
    """Handles solar panel detection using spectral indices."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_mask(self, tiff_array: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask for likely water-based solar panels.
        
        Args:
            tiff_array: Multispectral image array with shape (H, W, bands)
            
        Returns:
            Binary mask array
        """
        bands = self.config.bands
        
        # Extract bands
        blue = tiff_array[:, :, bands['blue']]
        green = tiff_array[:, :, bands['green']]
        red = tiff_array[:, :, bands['red']]
        nir = tiff_array[:, :, bands['nir']]
        swir_1 = tiff_array[:, :, bands['swir_1']]
        swir_2 = tiff_array[:, :, bands['swir_2']]

        # Calculate spectral indices
        bi_green = (blue - green) / (blue + green + 1e-8) > -0.07
        bi_red = (blue - red) / (blue + red + 1e-8) > -0.05
        ndbi = (swir_1 - nir) / (swir_1 + nir + 1e-8) > 0.02
        nsdsi = (swir_1 - swir_2) / (swir_1 + swir_2 + 1e-8) > 0.12

        # Combine conditions
        mask = (
            bi_green.astype(int) &
            (swir_1 > 0.07).astype(int) &
            bi_red.astype(int) &
            ndbi.astype(int) &
            nsdsi.astype(int)
        )

        return mask.astype(bool)


class DataFetcher:
    """Main class for fetching satellite data."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.processor = SentinelProcessor(self.config)
        
        if not EarthEngineManager.initialize(self.config.project_id):
            raise RuntimeError("Failed to initialize Google Earth Engine")

    def get_annual_composite_data(self, lon: float, lat: float, output_folder: str,
                                start_year: int, end_year: int, 
                                image_type: str = 'multispectral_tif') -> List[str]:
        """
        Fetches annual median composite images for a single coordinate.
        
        Args:
            lon, lat: Coordinates
            output_folder: Output directory
            start_year, end_year: Year range
            image_type: Type of image to download
            
        Returns:
            List of downloaded file paths
        """
        point_name = f"{lat:.4f}_{lon:.4f}"
        print(f"Processing point: {point_name} for years {start_year} to {end_year}")

        os.makedirs(output_folder, exist_ok=True)
        downloaded_paths = []
        region = self.processor.create_region(lon, lat)

        for year in range(start_year, end_year + 1):
            print(f"  Fetching data for year: {year}")
            
            composite = self.processor.get_annual_composite(lon, lat, year)
            if composite is None:
                continue

            base_filename = f"{point_name}_{year}"
            success = False

            if image_type == 'rgb_png':
                output_path = os.path.join(output_folder, f"{base_filename}_rgb.png")
                success = self.processor.download_rgb_png(composite, region, output_path)
                
            elif image_type == 'multispectral_tif':
                output_path = os.path.join(output_folder, f"{base_filename}_multispectral.tif")
                success = self.processor.download_multispectral_tif(composite, region, output_path)
                
                if success:
                    downloaded_paths.append(output_path)
                    
                    # Also download water mask
                    water_path = os.path.join(output_folder, f"{base_filename}_watermask.tif")
                    if self.processor.download_water_mask(region, water_path):
                        downloaded_paths.append(water_path)
                        
            elif image_type == 'torch_tensor':
                output_path = os.path.join(output_folder, f"{base_filename}.pt")
                success = self.processor.download_torch_tensor(
                    composite, region, output_path, lat, lon, year
                )
            else:
                raise ValueError(f"Invalid image_type: {image_type}")

            if success and image_type != 'multispectral_tif':  # Already added above
                downloaded_paths.append(output_path)
                print(f"    Saved {image_type} for {point_name} year {year}")

        return downloaded_paths

    def fetch_images_from_json(self, json_file_path: str, output_base_folder: str = 'fetched_images',
                             image_type: str = 'multispectral_tif', max_workers: int = 5,
                             start_year: int = 2020, end_year: int = 2024) -> List[Tuple]:
        """
        Fetches images for multiple coordinates from JSON file.
        
        Args:
            json_file_path: Path to JSON file with coordinates
            output_base_folder: Base output directory
            image_type: Type of images to download
            max_workers: Number of concurrent workers
            start_year, end_year: Year range
            
        Returns:
            List of (location_dict, result_path) tuples
        """
        os.makedirs(output_base_folder, exist_ok=True)

        with open(json_file_path, 'r') as f:
            locations = json.load(f)

        if not self._validate_locations(locations):
            raise ValueError("Invalid JSON format. Expected list of dicts with 'lon' and 'lat' keys.")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_coords = {
                executor.submit(
                    self.get_annual_composite_data, 
                    loc['lon'], loc['lat'], output_base_folder, 
                    start_year, end_year, image_type
                ): loc for loc in locations
            }

            for future in as_completed(future_to_coords):
                loc = future_to_coords[future]
                try:
                    result_paths = future.result()
                    if result_paths:
                        for path in result_paths:
                            results.append((loc, path))
                    else:
                        results.append((loc, "No images found"))
                except Exception as exc:
                    print(f"Location {loc['lat']},{loc['lon']} generated an exception: {exc}")
                    results.append((loc, "Error"))

        self._print_summary(results)
        return results

    @staticmethod
    def _validate_locations(locations: List) -> bool:
        """Validate location data format."""
        return (isinstance(locations, list) and 
                all(isinstance(loc, dict) and 'lon' in loc and 'lat' in loc 
                    for loc in locations))

    @staticmethod
    def _print_summary(results: List[Tuple]):
        """Print download summary."""
        print("\n--- Fetching Summary ---")
        for loc, status in results:
            print(f"Location ({loc['lat']:.4f}, {loc['lon']:.4f}): {status}")


class Visualizer:
    """Handles visualization of satellite data and analysis results."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.mask_generator = SolarPanelMask(self.config)

    def plot_example(self, locations: List[Dict], years: List[str], 
                    data_folder: str, output_path: str = 'example_segmentation.pdf'):
        """
        Plot example segmentation results for multiple locations and years.
        
        Args:
            locations: List of location dictionaries
            years: List of years to analyze
            data_folder: Folder containing downloaded data
            output_path: Output path for the plot
        """
        num_images = len(locations)
        n_cols = 2
        n_rows = math.ceil(num_images / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4, 6.4)) # Latex fig 6.4 x 6.4
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, loc in enumerate(locations):
            if i >= len(axes):
                break
                
            ax = axes[i]
            lat, lon = loc['lat'], loc['lon']
            title = f"Lat: {lat:.2f} Lon: {lon:.2f}"

            try:
                overlay = self._create_overlay(lat, lon, years, data_folder)
                if overlay is not None:
                    ax.imshow(overlay)
                    ax.set_title(title, fontsize=12)
                else:
                    ax.set_title(f"{title}\n(No Data)", fontsize=12)
            except Exception as e:
                print(f"Error processing location {lat}, {lon}: {e}")
                ax.set_title(f"{title}\n(Error)", fontsize=12)
            
            ax.axis('off')

        # Hide unused axes
        for j in range(len(locations), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _create_overlay(self, lat: float, lon: float, years: List[str], 
                       data_folder: str) -> Optional[np.ndarray]:
        """Create overlay image with solar panel detection."""
        yearly_masks = []

        for year in years:
            img_path = os.path.join(
                data_folder, f"{lat:.4f}_{lon:.4f}_{year}_multispectral.tif"
            )
            water_path = os.path.join(
                data_folder, f"{lat:.4f}_{lon:.4f}_{year}_watermask.tif"
            )

            if not os.path.exists(img_path):
                continue

            img = tifffile.imread(img_path)
            if img.shape != (256, 256, 12):
                print(f"Unexpected image shape: {img.shape}")
                continue

            # Load water mask
            if os.path.exists(water_path):
                water_mask = tifffile.imread(water_path) > 0
            else:
                water_mask = np.zeros(img.shape[:2], dtype=bool)

            # Get spectral mask and combine with water mask
            spectral_mask = self.mask_generator.get_mask(img)
            combined_mask = spectral_mask & water_mask
            yearly_masks.append(combined_mask)

        if not yearly_masks:
            return None

        # Combine yearly masks with consistency threshold
        mask_stack = np.stack(yearly_masks, axis=0)
        mask_mean = mask_stack.mean(axis=0)
        consistent_mask = mask_mean >= 0.1  # Adjust threshold as needed

        # Create RGB overlay
        latest_img_path = os.path.join(
            data_folder, f"{lat:.4f}_{lon:.4f}_{years[-1]}_multispectral.tif"
        )
        
        if os.path.exists(latest_img_path):
            img = tifffile.imread(latest_img_path)
            rgb_display = img[:, :, [3, 2, 1]]  # RGB bands
            rgb_display = np.clip(rgb_display, 0, 0.3) / 0.3
            
            overlay = rgb_display.copy()
            overlay[consistent_mask] = [1.0, 0.0, 0.0]  # Red highlight
            return overlay

        return None


# Convenience functions for backward compatibility
def fetch_images_from_json(json_file_path: str, **kwargs) -> List[Tuple]:
    """Convenience wrapper for DataFetcher.fetch_images_from_json."""
    fetcher = DataFetcher()
    return fetcher.fetch_images_from_json(json_file_path, **kwargs)


def plot_example(locations: Union[str, List[Dict]], years: List[str], 
                data_folder: str = 'downloaded_s2_annual_composites', **kwargs):
    """Convenience wrapper for Visualizer.plot_example."""
    if isinstance(locations, str):
        with open(locations, 'r') as f:
            locations = json.load(f)
    
    visualizer = Visualizer()
    visualizer.plot_example(locations, years, data_folder, **kwargs)