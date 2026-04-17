"""
Test script for downloading Sentinel-2 data for Mangalore coastal area.
"""

from pathlib import Path
import sys
from datetime import datetime
import logging

# Add project source to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from download_sentinel_aws import find_sentinel_scenes, download_bands

def download_mangalore_data():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create required directories
    try:
        Path("data").mkdir(exist_ok=True)
        Path("data/sentinel").mkdir(exist_ok=True)
        logger.info("Created required directories")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        return 1
    
    # Mangalore area parameters
    TILE_ID = "43PGP"  # Tile covering Mangalore coastal area
    
    # Study area details
    print("\n=== Mangalore Coastal Study Area ===")
    print("Location: New Mangalore Port and surrounding waters")
    print("Coordinates: 12.95° N, 74.80° E")
    print("Tile ID: 43PGP")
    print("Date of Interest: 2025-11-04")
    print("Cloud Cover: 30%")
    print("Area Features:")
    print("- Port infrastructure")
    print("- Coastal waters")
    print("- Variable depths")
    print("- Clear water conditions")
    
    try:
        # Search for scenes with very low cloud cover for better quality
        print("\nSearching for clear scenes over Mangalore...")
        scenes = find_sentinel_scenes(
            tile_id=TILE_ID,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31),
            max_cloud_cover=10.0  # Stricter cloud cover requirement
        )
        
        if not scenes:
            print("\nNo clear scenes found. Trying with relaxed cloud cover...")
            scenes = find_sentinel_scenes(
                tile_id=TILE_ID,
                start_date=datetime(2025, 1, 1),
                end_date=datetime(2025, 12, 31),
                max_cloud_cover=20.0
            )
        
        if not scenes:
            print("\nNo suitable scenes found. Please try different dates.")
            return
        
        # Sort scenes by cloud cover
        scenes.sort(key=lambda x: x['cloud_cover'])
        
        print(f"\nFound {len(scenes)} suitable scenes:")
        for i, scene in enumerate(scenes, 1):
            print(f"\n{i}. Scene Details:")
            print(f"   Date: {scene['date']}")
            print(f"   Cloud Cover: {scene['cloud_cover']:.1f}%")
        
        # Automatically select the clearest scene
        best_scene = scenes[0]
        print(f"\nSelecting clearest scene from {best_scene['date']} with {best_scene['cloud_cover']:.1f}% cloud cover")
        
        # Create output directory with date
        date_str = datetime.strptime(best_scene['date'], "%Y-%m-%d").strftime("%Y%m%d")
        output_dir = Path("data/sentinel") / f"mangalore_{date_str}"
        
        print(f"\nDownloading bands to {output_dir}...")
        band_paths = download_bands(
            scene_path=best_scene['aws_path'],
            output_dir=output_dir,
            bands=['B02', 'B03', 'B04', 'B08'],
            resolution=10
        )
        
        print("\nDownload completed! Files saved:")
        for band, path in band_paths.items():
            print(f"- {band}: {path}")
        
        print("\nNext steps:")
        print("1. Use these files in the SDB processing notebook")
        print("2. Focus on the port area and nearby waters")
        print("3. Compare results with known port depths")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(download_mangalore_data())