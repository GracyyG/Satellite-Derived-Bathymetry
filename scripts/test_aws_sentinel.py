"""
Test script for accessing Sentinel-2 data from AWS Open Data.
"""

from pathlib import Path
import sys
from datetime import datetime
import json

# Add project source to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from download_sentinel_aws import find_sentinel_scenes, download_bands
import logging

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test parameters
    tile_id = input("Enter Sentinel-2 tile ID (e.g., '10SEG'): ")
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    try:
        # Search for scenes
        print(f"\nSearching for scenes in tile {tile_id}...")
        scenes = find_sentinel_scenes(
            tile_id=tile_id,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=20.0
        )
        
        if not scenes:
            print("\nNo scenes found matching criteria.")
            return
        
        print(f"\nFound {len(scenes)} scenes:")
        for i, scene in enumerate(scenes, 1):
            print(f"\n{i}. Scene Details:")
            print(f"   Date: {scene['date']}")
            print(f"   Cloud Cover: {scene['cloud_cover']}%")
            print(f"   AWS Path: {scene['aws_path']}")
        
        # Ask user which scene to download
        if len(scenes) > 0:
            scene_idx = int(input(f"\nEnter scene number to download (1-{len(scenes)}): ")) - 1
            if 0 <= scene_idx < len(scenes):
                scene = scenes[scene_idx]
                output_dir = Path("data/sentinel")
                
                print(f"\nDownloading bands from scene {scene_idx + 1}...")
                band_paths = download_bands(
                    scene_path=scene['aws_path'],
                    output_dir=output_dir,
                    bands=['B02', 'B03', 'B04', 'B08'],
                    resolution=10
                )
                
                print("\nDownload completed! Band locations:")
                for band, path in band_paths.items():
                    print(f"{band}: {path}")
                    
                # Save metadata
                metadata_path = output_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(scene['metadata'], f, indent=2)
                print(f"\nMetadata saved to {metadata_path}")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())