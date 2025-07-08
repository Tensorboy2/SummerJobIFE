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
import scipy.ndimage


@dataclass
class Config:
    """Configuration class for satellite data processing."""
    project_id: str = 'summer-job-ife'
    buffer_degrees: float = 0.012
    image_dimensions: str = '256x256'
    cloud_threshold: float = 35.0
    water_occurrence_threshold: float = 50.0
    crs: str = 'EPSG:4326'
    
    # Sentinel-1 water detection threshold (dB)
    s1_water_threshold: float = -15.0
    
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
    """Handles Sentinel-1 and Sentinel-2 data processing and downloading."""
    
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

    def get_s1_annual_composite(self, lon: float, lat: float, year: int) -> Optional[ee.Image]:
        """Get annual median composite for Sentinel-1 data."""
        region = self.create_region(lon, lat)
        
        dataset = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(region)
            .filterDate(f'{year}-01-01', f'{year}-12-31')
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select(['VV', 'VH'])
        )

        count = dataset.size().getInfo()
        if count == 0:
            print(f"    Warning: No Sentinel-1 images found for {lat:.4f}_{lon:.4f} in year {year}")
            return None

        try:
            composite = dataset.median()
            _ = composite.getInfo()  # Test if composite is valid
            return composite
        except Exception as e:
            print(f"    Warning: Failed to create S1 composite for {lat:.4f}_{lon:.4f} in {year}: {e}")
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

    def download_s1_tif(self, composite: ee.Image, region: ee.Geometry, 
                       output_path: str) -> bool:
        """Download Sentinel-1 GeoTIFF image."""
        download_params = {
            'crs': self.config.crs,
            'region': region.getInfo()['coordinates'],
            'format': 'GEO_TIFF',
            'dimensions': self.config.image_dimensions
        }

        try:
            download_url = composite.getDownloadURL(download_params)
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"    Failed to download Sentinel-1 TIFF: {e}")
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

    def download_s1_water_mask(self, composite: ee.Image, region: ee.Geometry, 
                              output_path: str) -> bool:
        """Download Sentinel-1 based water mask."""
        try:
            # Use VV polarization for water detection
            vv_band = composite.select('VV')
            
            # Create water mask using threshold (water has low backscatter)
            water_mask = vv_band.lt(self.config.s1_water_threshold)

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
            print(f"    Failed to download S1 water mask: {e}")
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

class WaterMaskProcessor:
    """Handles water mask processing from multiple sources."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_s1_water_mask(self, s1_array: np.ndarray, 
                         polarization: str = 'VV') -> np.ndarray:
        """
        Create water mask from Sentinel-1 SAR data.
        
        Args:
            s1_array: Sentinel-1 image array with shape (H, W, bands) or (H, W)
            polarization: Polarization to use ('VV' or 'VH')
            
        Returns:
            Binary water mask (1 for water, 0 for non-water)
        """
        if s1_array.ndim == 3:
            # Assume VV is first band, VH is second
            if polarization == 'VV':
                sar_band = s1_array[:, :, 0]
            elif polarization == 'VH':
                sar_band = s1_array[:, :, 1]
            else:
                raise ValueError("Polarization must be 'VV' or 'VH'")
        else:
            # Single band array
            sar_band = s1_array
        
        # Convert to dB if not already (check if values are in linear scale)
        if np.max(sar_band) > 1.0:
            # Assume linear scale, convert to dB
            sar_band_db = 10 * np.log10(sar_band + 1e-8)
        else:
            # Assume already in dB
            sar_band_db = sar_band
        
        # Create water mask using threshold
        water_mask = (sar_band_db < self.config.s1_water_threshold).astype(int)
        
        return water_mask
    
    def combine_water_masks(self, jrc_mask: np.ndarray, 
                           s1_mask: Optional[np.ndarray] = None,
                           method: str = 'union') -> np.ndarray:
        """
        Combine water masks from different sources.
        
        Args:
            jrc_mask: JRC Global Surface Water mask
            s1_mask: Sentinel-1 derived water mask
            method: Combination method ('union', 'intersection', 'jrc_only', 's1_only')
            
        Returns:
            Combined water mask
        """
        if s1_mask is None:
            return jrc_mask
        
        # Ensure both masks are binary
        jrc_binary = (jrc_mask > 0).astype(int)
        s1_binary = (s1_mask > 0).astype(int)
        
        if method == 'union':
            # Water if detected by either source
            combined = np.logical_or(jrc_binary, s1_binary).astype(int)
        elif method == 'intersection':
            # Water only if detected by both sources
            combined = np.logical_and(jrc_binary, s1_binary).astype(int)
        elif method == 'jrc_only':
            combined = jrc_binary
        elif method == 's1_only':
            combined = s1_binary
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return combined

class DataFetcher:
    """Main class for fetching satellite data."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.processor = SentinelProcessor(self.config)
        self.water_processor = WaterMaskProcessor(self.config)
        
        if not EarthEngineManager.initialize(self.config.project_id):
            raise RuntimeError("Failed to initialize Google Earth Engine")

    def get_annual_composite_data(self, lon: float, lat: float, output_folder: str,
                                start_year: int, end_year: int, 
                                image_type: str = 'multispectral_tif',
                                include_s1: bool = True) -> List[str]:
        """
        Fetches annual median composite images for a single coordinate.
        
        Args:
            lon, lat: Coordinates
            output_folder: Output directory
            start_year, end_year: Year range
            image_type: Type of image to download
            include_s1: Whether to download Sentinel-1 data
            
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
            
            # Get Sentinel-2 composite
            s2_composite = self.processor.get_annual_composite(lon, lat, year)
            if s2_composite is None:
                continue

            # Get Sentinel-1 composite if requested
            s1_composite = None
            if include_s1:
                s1_composite = self.processor.get_s1_annual_composite(lon, lat, year)

            base_filename = f"{point_name}_{year}"
            success = False

            if image_type == 'rgb_png':
                output_path = os.path.join(output_folder, f"{base_filename}_rgb.png")
                success = self.processor.download_rgb_png(s2_composite, region, output_path)
                
            elif image_type == 'multispectral_tif':
                output_path = os.path.join(output_folder, f"{base_filename}_multispectral.tif")
                success = self.processor.download_multispectral_tif(s2_composite, region, output_path)
                
                if success:
                    downloaded_paths.append(output_path)
                    
                    # Download JRC water mask
                    water_path = os.path.join(output_folder, f"{base_filename}_watermask.tif")
                    if self.processor.download_water_mask(region, water_path):
                        downloaded_paths.append(water_path)
                    
                    # Download Sentinel-1 data and water mask if available
                    if s1_composite is not None:
                        s1_path = os.path.join(output_folder, f"{base_filename}_s1.tif")
                        if self.processor.download_s1_tif(s1_composite, region, s1_path):
                            downloaded_paths.append(s1_path)
                        
                        s1_water_path = os.path.join(output_folder, f"{base_filename}_s1_watermask.tif")
                        if self.processor.download_s1_water_mask(s1_composite, region, s1_water_path):
                            downloaded_paths.append(s1_water_path)
                        
            elif image_type == 'torch_tensor':
                output_path = os.path.join(output_folder, f"{base_filename}.pt")
                success = self.processor.download_torch_tensor(
                    s2_composite, region, output_path, lat, lon, year
                )
            else:
                raise ValueError(f"Invalid image_type: {image_type}")

            if success and image_type != 'multispectral_tif':  # Already added above
                downloaded_paths.append(output_path)
                print(f"    Saved {image_type} for {point_name} year {year}")

        return downloaded_paths

    def fetch_images_from_json(self, json_file_path: str, output_base_folder: str = 'fetched_images',
                             image_type: str = 'multispectral_tif', max_workers: int = 5,
                             start_year: int = 2020, end_year: int = 2024,
                             include_s1: bool = True) -> List[Tuple]:
        """
        Fetches images for multiple coordinates from JSON file.
        
        Args:
            json_file_path: Path to JSON file with coordinates
            output_base_folder: Base output directory
            image_type: Type of images to download
            max_workers: Number of concurrent workers
            start_year, end_year: Year range
            include_s1: Whether to download Sentinel-1 data
            
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
                    start_year, end_year, image_type, include_s1
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
        self.water_processor = WaterMaskProcessor(self.config)

    def plot_example(self, locations: List[Dict], years: List[str], 
                    data_folder: str, output_path: str = 'example_segmentation.pdf',
                    water_mask_method: str = 'union'):
        """
        For each location, show a single figure with subplots:
        - RGB
        - JRC water mask
        - S1 water mask
        - Combined water mask
        - Spectral mask
        - Final result (RGB with mask overlay)
        """
        for loc in locations:
            lat, lon = loc['lat'], loc['lon']
            title = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
            try:
                yearly_data = {}
                available_years = []
                for year in years:
                    img_path = os.path.join(
                        data_folder, f"{lat:.4f}_{lon:.4f}_{year}_multispectral.tif"
                    )
                    water_path = os.path.join(
                        data_folder, f"{lat:.4f}_{lon:.4f}_{year}_watermask.tif"
                    )
                    s1_water_path = os.path.join(
                        data_folder, f"{lat:.4f}_{lon:.4f}_{year}_s1_watermask.tif"
                    )
                    
                    if not os.path.exists(img_path):
                        continue
                        
                    img = tifffile.imread(img_path)
                    if img.shape != (256, 256, 12):
                        print(f"Unexpected image shape for {year}: {img.shape}")
                        continue
                        
                    # Load masks
                    jrc_mask = tifffile.imread(water_path) > 0 if os.path.exists(water_path) else np.zeros(img.shape[:2], dtype=bool)
                    s1_mask = tifffile.imread(s1_water_path) > 0 if os.path.exists(s1_water_path) else np.zeros(img.shape[:2], dtype=bool)
                    
                    # Get spectral mask
                    spectral_mask = self.mask_generator.get_mask(img)
                    
                    yearly_data[year] = {
                        'img': img,
                        'jrc_mask': jrc_mask,
                        's1_mask': s1_mask,
                        'spectral_mask': spectral_mask
                    }
                    available_years.append(year)
                
                if not available_years:
                    print(f"No valid images found for {title}")
                    return
                
                print(f"Processing {len(available_years)} years for {title}: {available_years}")
                
                # Perform consistency analysis
                spectral_masks = np.stack([yearly_data[year]['spectral_mask'] for year in available_years])
                jrc_masks = np.stack([yearly_data[year]['jrc_mask'] for year in available_years])
                
                # Calculate consistency metrics
                spectral_consistency = np.mean(spectral_masks, axis=0)  # Fraction of years with spectral water detection
                jrc_consistency = np.mean(jrc_masks.astype(float), axis=0)  # Fraction of years with JRC water
                
                # Agreement between spectral and JRC masks across years
                agreement_mask = np.mean(spectral_masks == jrc_masks, axis=0)  # Fraction of years with agreement
                
                # Create consensus masks based on consistency thresholds
                consistent_spectral = spectral_consistency >= 0.5  # Water detected in ≥50% of years
                consistent_jrc = jrc_consistency >= 0.5  # JRC water in ≥50% of years
                high_agreement = agreement_mask >= 0.7  # Agreement in ≥70% of years
                
                # Use the most recent year for final processing
                latest_year = years[-1]
                latest_data = yearly_data[latest_year]
                
                # Combine water masks using consistency information
                # Priority: consistent detections, then recent detections
                base_water_mask = consistent_spectral & consistent_jrc
                
                # Add recent detections where there's high agreement
                recent_spectral = latest_data['spectral_mask']
                recent_jrc = latest_data['jrc_mask']
                recent_agreement = recent_spectral == recent_jrc
                
                enhanced_water_mask = base_water_mask | (recent_agreement & (recent_spectral | recent_jrc))
                
                # Final masking with S1 from the latest year
                s1_mask = latest_data['s1_mask']
                # Combine all masks: (consistent water OR recent agreed water) AND S1 confirmation
                final_mask = enhanced_water_mask & np.logical_not(s1_mask)
                # Alternative: Use S1 as validation rather than strict intersection
                final_mask = enhanced_water_mask & (np.logical_not(s1_mask) & (spectral_consistency > 0.7))
                # --- Morphological opening to remove single-pixel noise ---
                # structure = np.ones((3, 3), dtype=bool)
                # final_mask_clean = scipy.ndimage.binary_closing(final_mask, structure=structure)
                # labeled_mask, num_features = scipy.ndimage.label(final_mask)
                # Prepare visualization
                rgb_display = latest_data['img'][:, :, [3, 2, 1]]
                rgb_display = np.clip(rgb_display, 0, 0.3) / 0.3
                
                # Create overlay
                overlay = rgb_display.copy()
                overlay[enhanced_water_mask] = [1.0, 0.0, 0.0]  # Red for final water mask
                
                # Plot comprehensive results
                fig, axs = plt.subplots(2, 5, figsize=(12, 6))
                
                # Top row: Individual masks and consistency
                axs[0,0].imshow(rgb_display)
                axs[0,0].set_title(f'RGB ({latest_year})')
                
                axs[0,1].imshow(spectral_consistency, cmap='Greens', vmin=0, vmax=1)
                axs[0,1].set_title('Spectral Consistency\n(Fraction of Years)')
                
                axs[0,2].imshow(jrc_consistency, cmap='Blues', vmin=0, vmax=1)
                axs[0,2].set_title('JRC Consistency\n(Fraction of Years)')
                
                axs[0,3].imshow(agreement_mask, cmap='Purples', vmin=0, vmax=1)
                axs[0,3].set_title('Spectral-JRC Agreement\n(Fraction of Years)')
                
                axs[0,4].imshow(s1_mask, cmap='Oranges')
                axs[0,4].set_title(f'S1 Water Mask ({latest_year})')
                
                # Bottom row: Processing steps and final result
                axs[1,0].imshow(consistent_spectral, cmap='Greens')
                axs[1,0].set_title('Consistent Spectral\n(≥50% years)')
                
                axs[1,1].imshow(consistent_jrc, cmap='Blues')
                axs[1,1].set_title('Consistent JRC\n(≥50% years)')
                
                axs[1,2].imshow(enhanced_water_mask, cmap='Purples')
                axs[1,2].set_title('Enhanced Water Mask\n(Consistent + Recent)')
                
                axs[1,3].imshow(final_mask, cmap='Reds')
                axs[1,3].set_title('Final Mask\n(Enhanced ∩ S1, Cleaned)')
                
                axs[1,4].imshow(overlay)
                axs[1,4].set_title('Final Result Overlay')
                
                for ax in axs.flat:
                    ax.axis('off')
                
                # Add colorbar for consistency plots
                # for i in range(1, 4):
                #     cbar = plt.colorbar(axs[0,i].images[0], ax=axs[0,i], shrink=0.6)
                #     cbar.set_label('Fraction')
                
                fig.suptitle(f'{title} - Multi-Year Consistency Analysis ({len(available_years)} years)', fontsize=16)
                plt.tight_layout()
                plt.show()
                
                # Print summary statistics
                print(f"\nSummary for {title}:")
                print(f"  Available years: {len(available_years)}")
                print(f"  Spectral water pixels (consistent): {np.sum(consistent_spectral)}")
                print(f"  JRC water pixels (consistent): {np.sum(consistent_jrc)}")
                print(f"  S1 water pixels (latest): {np.sum(s1_mask)}")
                print(f"  Final water pixels: {np.sum(final_mask)}")
                print(f"  Agreement rate (spectral-JRC): {np.mean(agreement_mask):.2%}")
                
            except Exception as e:
                print(f"Error processing location {lat}, {lon}: {e}")
                import traceback
                traceback.print_exc()
                continue


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