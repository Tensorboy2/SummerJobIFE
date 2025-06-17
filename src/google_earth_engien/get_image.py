import ee
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
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

def get_single_image_data(lon, lat, output_folder, image_type='multispectral_tif'):
    """
    Fetches image data for a single lon/lat pair from GEE.
    Supports fetching either visualization PNGs or raw multispectral GeoTIFFs.
    """
    point_name = f"{lat:.4f}_{lon:.4f}"
    print(f"Processing point: {point_name}")

    try:
        buffer_deg = 0.012 # Roughly 1.3km on each side, so 2.6km total extent
        region = ee.Geometry.BBox(lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)

        # Load Sentinel-2 data
        dataset = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate('2020-01-01', '2020-12-31') 
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)) # Cloud filter
            .map(mask_s2_clouds)
        )

        image = dataset.median() # Use median instead of mean for better pixel value representation

        # Check if an image was found
        try:
            # Try to get info to check if image is not empty
            _ = image.getInfo()
        except Exception as e:
            print(f"Warning: No valid image found for {point_name}. Error: {e}")
            return None # Return None if no image found

        if image_type == 'rgb_png':
            # Visualization parameters for PNG thumbnail 
            vis_params = {
                'min': 0.0,
                'max': 0.3,
                'bands': ['B4', 'B3', 'B2'],
            }
            thumb_url = image.getThumbURL({
                'dimensions': 256, # Match your desired output size, e.g., 256 for GloSoFarID
                'region': region,
                'format': 'png',
                'min': vis_params['min'],
                'max': vis_params['max'],
                'bands': vis_params['bands']
            })

            response = requests.get(thumb_url)
            if response.status_code == 200:
                img_pil = Image.open(BytesIO(response.content))
                img_path = os.path.join(output_folder, f"{point_name}_rgb.png")
                img_pil.save(img_path)
                print(f"Saved RGB PNG for {point_name} to {img_path}")
                return img_path
            else:
                print(f"Failed to download RGB PNG for {point_name}: Status {response.status_code}")
                return None

        elif image_type == 'multispectral_tif':
            # Bands:
            bands_to_download = [
                'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'
            ]

            # Ensure the image has the selected bands
            image_selected_bands = image.select(bands_to_download)

            # Define download parameters for GeoTIFF
            download_params = {
                'crs': 'EPSG:4326', # WGS84 geographic coordinate system
                'region': region.getInfo()['coordinates'], # GEE expects raw coordinates
                'format': 'GEO_TIFF',
                'dimensions': '128x128' # Specify output dimensions directly (Suitable for Pytorch training)
            }
            download_url = image_selected_bands.getDownloadURL(download_params)

            # Download the GeoTIFF
            response = requests.get(download_url, stream=True)
            if response.status_code == 200:
                tif_path = os.path.join(output_folder, f"{point_name}_multispectral.tif")
                with open(tif_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Saved multispectral GeoTIFF for {point_name} to {tif_path}")
                return tif_path
            else:
                print(f"Failed to download multispectral GeoTIFF for {point_name}: Status {response.status_code}")
                # Print response content for more detailed error from GEE
                return None
        else:
            raise ValueError("Invalid image_type. Choose 'rgb_png' or 'multispectral_tif'.")

    except Exception as e:
        print(f"An error occurred while fetching image for {point_name}: {e}")
        return None

def fetch_images_from_json(json_file_path, output_base_folder='fetched_images', image_type='multispectral_tif', max_workers=5):
    """
    Fetches images for multiple lon/lat pairs from a JSON file concurrently.
    """
    os.makedirs(output_base_folder, exist_ok=True)
    
    with open(json_file_path, 'r') as f:
        locations = json.load(f)

    # Validate JSON structure (expecting a list of dictionaries with 'lon' and 'lat')
    if not isinstance(locations, list) or not all(isinstance(loc, dict) and 'lon' in loc and 'lat' in loc for loc in locations):
        raise ValueError("JSON file must contain a list of dictionaries, each with 'lon' and 'lat' keys.")

    results = []
    # Use ThreadPoolExecutor for concurrent HTTP requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_coords = {
            executor.submit(get_single_image_data, loc['lon'], loc['lat'], output_base_folder, image_type): loc
            for loc in locations
        }

        # Process results as they complete
        for future in as_completed(future_to_coords):
            loc = future_to_coords[future]
            try:
                result_path = future.result()
                if result_path:
                    results.append((loc, result_path))
                else:
                    results.append((loc, "Failed"))
            except Exception as exc:
                print(f"Location {loc['lat']},{loc['lon']} generated an exception: {exc}")
                results.append((loc, "Error"))
    
    print("\n--- Fetching Summary ---")
    for loc, status in results:
        print(f"Location ({loc['lat']:.4f}, {loc['lon']:.4f}): {status}")

    return results

# --- Example Usage ---
if __name__ == "__main__":
    fetched_data_paths = fetch_images_from_json(
        os.path.join(root,'coordinates.json'), 
        output_base_folder=os.path.join(root,'downloaded_s2_images'), 
        image_type='multispectral_tif', # Change to 'rgb_png' if you want PNGs
        max_workers=5 # Adjust based on your connection and GEE's rate limits
    )

    print("\nAll download tasks completed.")