"""
main.py

Main execution script for satellite data fetching and analysis.
"""
import os
import json
from utils import DataFetcher, Visualizer, Config


def main():
    """Main execution function."""
    # Configuration
    root = os.path.dirname(__file__)
    coordinates_file = os.path.join(root, 'shark.json')
    output_folder = os.path.join(root, 'downloaded_s2_annual_composites')
    
    # Parameters
    start_year = 2025  # Sentinel-2 data typically starts mid-2015
    end_year = 2025    # Up to the current full year
    analysis_years = ['2025']
    
    # Create custom config if needed
    config = Config(
        project_id='summer-job-ife',
        buffer_degrees=0.010,
        cloud_threshold=35.0,
        water_occurrence_threshold=10.0,
    )
    
    # Check if data already exists
    if os.path.exists(output_folder) and os.listdir(output_folder):
        print(f"Data folder {output_folder} already exists with files.")
        user_input = input("Skip download and proceed to visualization? (y/n): ").lower()
        
        if user_input != 'y':
            download_data(coordinates_file, output_folder, config, start_year, end_year)
    else:
        download_data(coordinates_file, output_folder, config, start_year, end_year)
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(coordinates_file, analysis_years, output_folder)
    
    print("\nProcessing completed successfully!")


def download_data(coordinates_file: str, output_folder: str, config: Config,
                 start_year: int, end_year: int):
    """Download satellite data."""
    if not os.path.exists(coordinates_file):
        print(f"Error: Coordinates file {coordinates_file} not found!")
        print("Please create a JSON file with coordinates in the following format:")
        print('[{"lat": 40.7128, "lon": -74.0060}, {"lat": 51.5074, "lon": -0.1278}]')
        return
    
    print(f"Starting download from {start_year} to {end_year}...")
    
    try:
        fetcher = DataFetcher(config)
        fetched_data_paths = fetcher.fetch_images_from_json(
            coordinates_file,
            output_base_folder=output_folder,
            image_type='multispectral_tif',
            max_workers=8,  # Adjust based on your connection and GEE's rate limits
            start_year=start_year,
            end_year=end_year
        )
        
        print(f"\nDownloaded {len(fetched_data_paths)} files total.")
        
    except Exception as e:
        print(f"Error during download: {e}")
        raise


def create_visualization(coordinates_file: str, years: list, data_folder: str):
    """Create visualization of the analysis results."""
    try:
        with open(coordinates_file, 'r') as f:
            locations = json.load(f)
        visualizer = Visualizer()
        visualizer.plot_example(
            locations=locations,
            years=years,
            data_folder=data_folder,
            output_path='example_segmentation.pdf',
            water_mask_method='union',
        )
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise

if __name__ == '__main__':
    main()