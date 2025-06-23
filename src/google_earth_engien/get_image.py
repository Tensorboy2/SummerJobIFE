import ee
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tifffile
import torch
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime

root = os.path.dirname(__file__)
folder = 'downloaded_s2_images'

try:
    ee.Initialize(project='summer-job-ife')
    print("Earth Engine initialized successfully.")
except ee.EEException as e:
    print(f"Earth Engine initialization failed: {e}")
    print("Please run 'ee.Authenticate()' in your terminal/notebook if you haven't.")
    exit() # Exit if EE cannot be initialized

def mask_s2_clouds(image):
    """Masks clouds in Sentinel-2 image using the QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000) # Reflectance values are 0-10000

def get_annual_composite_data(lon, lat, output_folder, start_year, end_year, image_type='multispectral_tif'):
    """
    Fetches one annual median composite image for a single lon/lat pair for each year
    within the specified range from GEE.
    """
    point_name = f"{lat:.4f}_{lon:.4f}"
    print(f"Processing point: {point_name} for years {start_year} to {end_year}")

    downloaded_paths = []
    buffer_deg = 0.012
    region = ee.Geometry.BBox(lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)

    for year in range(start_year, end_year + 1):
        print(f"  Fetching data for year: {year}")
        try:
            # Filter for the current year
            dataset_yearly = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(region)
                .filterDate(f'{year}-01-01', f'{year}-12-31')
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)) # Cloud filter
                .map(mask_s2_clouds)
            )

            # Check if there are any images for the year before taking the median
            count = dataset_yearly.size().getInfo()
            if count == 0:
                print(f"    Warning: No clear-sky images found for {point_name} in year {year}. Skipping.")
                continue

            # Compute the median composite for the year
            annual_composite = dataset_yearly.median()

            # Check if the composite image is valid (not empty)
            try:
                _ = annual_composite.getInfo()
            except Exception as e:
                print(f"    Warning: Median composite for {point_name} in year {year} is empty. Error: {e}. Skipping.")
                continue

            image_filename_base = f"{point_name}_{year}"

            if image_type == 'rgb_png':
                vis_params = {
                    'min': 0.0,
                    'max': 0.3,
                    'bands': ['B4', 'B3', 'B2'],
                }
                thumb_url = annual_composite.getThumbURL({
                    'dimensions': 256,
                    'region': region,
                    'format': 'png',
                    'min': vis_params['min'],
                    'max': vis_params['max'],
                    'bands': vis_params['bands']
                })

                response = requests.get(thumb_url)
                if response.status_code == 200:
                    img_pil = Image.open(BytesIO(response.content))
                    img_path = os.path.join(output_folder, f"{image_filename_base}_rgb.png")
                    img_pil.save(img_path)
                    print(f"    Saved RGB PNG for {point_name} for year {year} to {img_path}")
                    downloaded_paths.append(img_path)
                else:
                    print(f"    Failed to download RGB PNG for {point_name} for year {year}: Status {response.status_code}")

            elif image_type == 'multispectral_tif':
                bands_to_download = [
                    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'
                ]
                image_selected_bands = annual_composite.select(bands_to_download)

                download_params = {
                    'crs': 'EPSG:4326',
                    'region': region.getInfo()['coordinates'],
                    'format': 'GEO_TIFF',
                    'dimensions': '256x256'
                }
                download_url = image_selected_bands.getDownloadURL(download_params)

                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    tif_path = os.path.join(output_folder, f"{image_filename_base}_multispectral.tif")
                    with open(tif_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"    Saved multispectral GeoTIFF for {point_name} for year {year} to {tif_path}")
                    downloaded_paths.append(tif_path)
                else:
                    print(f"    Failed to download multispectral GeoTIFF for {point_name} for year {year}: Status {response.status_code}")

            elif image_type == 'torch_tensor':
                bands_to_download = [
                    'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                    'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'
                ]
                image_selected_bands = annual_composite.select(bands_to_download)

                download_params = {
                    'crs': 'EPSG:4326',
                    'region': region.getInfo()['coordinates'],
                    'format': 'GEO_TIFF',
                    'dimensions': '256x256'
                }

                download_url = image_selected_bands.getDownloadURL(download_params)
                response = requests.get(download_url, stream=True)

                if response.status_code == 200:
                    tif_path_temp = os.path.join(output_folder, f"{image_filename_base}_temp.tif")
                    with open(tif_path_temp, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    img_np = tifffile.imread(tif_path_temp)
                    if img_np.ndim == 3 and img_np.shape[0] < img_np.shape[-1]:
                        img_np = np.moveaxis(img_np, -1, 0)

                    img_tensor = torch.tensor(img_np, dtype=torch.float32)
                    pt_path = os.path.join(output_folder, f"{image_filename_base}.pt")
                    torch.save({'image': img_tensor, 'coords': (lat, lon), 'year': year}, pt_path)
                    os.remove(tif_path_temp)

                    print(f"    Saved PyTorch tensor for {point_name} for year {year} to {pt_path}")
                    downloaded_paths.append(pt_path)
                else:
                    print(f"    Failed to download GeoTIFF for {point_name} for year {year}: Status {response.status_code}")
            else:
                raise ValueError("Invalid image_type. Choose 'rgb_png', 'multispectral_tif', or 'torch_tensor'.")

        except Exception as e:
            print(f"An error occurred while fetching image for {point_name} in year {year}: {e}")

    return downloaded_paths


def fetch_images_from_json(json_file_path, output_base_folder='fetched_images', image_type='multispectral_tif', max_workers=5, start_year=2020, end_year=2024):
    """
    Fetches annual median composite images for multiple lon/lat pairs from a JSON file concurrently.
    """
    os.makedirs(output_base_folder, exist_ok=True)

    with open(json_file_path, 'r') as f:
        locations = json.load(f)

    if not isinstance(locations, list) or not all(isinstance(loc, dict) and 'lon' in loc and 'lat' in loc for loc in locations):
        raise ValueError("JSON file must contain a list of dictionaries, each with 'lon' and 'lat' keys.")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_coords = {
            executor.submit(get_annual_composite_data, loc['lon'], loc['lat'], output_base_folder, start_year, end_year, image_type): loc
            for loc in locations
        }

        for future in as_completed(future_to_coords):
            loc = future_to_coords[future]
            try:
                result_paths = future.result()
                if result_paths:
                    for path in result_paths:
                        results.append((loc, path))
                else:
                    results.append((loc, "No images found or download failed for any year"))
            except Exception as exc:
                print(f"Location {loc['lat']},{loc['lon']} generated an exception: {exc}")
                results.append((loc, "Error"))

    print("\n--- Fetching Summary ---")
    for loc, status in results:
        print(f"Location ({loc['lat']:.4f}, {loc['lon']:.4f}): {status}")

    return results

# --- Example Usage ---
if __name__ == "__main__":
    # Define the range of years you want annual composites for
    start_year_desired = 2018 # Sentinel-2 data typically starts mid-2015
    end_year_desired = 2024   # Up to the current full year

    fetched_data_paths = fetch_images_from_json(
        os.path.join(root, 'coordinates.json'),
        output_base_folder=os.path.join(root, 'downloaded_s2_annual_composites'), # New folder name for clarity
        image_type='multispectral_tif',
        max_workers=16, # Adjust based on your connection and GEE's rate limits
        start_year=start_year_desired,
        end_year=end_year_desired
    )

    print("\nAll annual composite download tasks completed.")