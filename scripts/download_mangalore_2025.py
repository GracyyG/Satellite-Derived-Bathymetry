"""
Download Sentinel-2 data for New Mangalore Port (2025-11-01)
Coordinates: 12.92°N, 74.82°E
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_mangalore_scene():
    try:
        # Create directories
        data_dir = Path("data/sentinel/mangalore")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure anonymous S3 access
        s3 = boto3.client(
            's3',
            config=Config(
                signature_version=UNSIGNED,
                retries={'max_attempts': 3},
                connect_timeout=10,
                read_timeout=30
            )
        )
        
        # Mangalore scene parameters - using verified 2023 data
        TILES = [
            # Primary tile with various date formats
            {
                'id': '43PGQ',
                'path': 'products/2023/10/S2B_MSIL2A_20231015T051649_N0509_R062_T43PGQ_20231015T070630'
            },
            {
                'id': '43PGQ',
                'path': 'tiles/43/P/GQ/2023/10/15'
            },
            {
                'id': '43PGP',
                'path': 'tiles/43/P/GP/2023/10/15'
            }
        ]
        BUCKET = "sentinel-s2-l2a"
        BANDS = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
        
        print("Accessing Sentinel-2 data...")
        print("Location: New Mangalore Port (12.92°N, 74.82°E)")
        print("Date: 2023-10-15 (Historical stable data)\n")
        
        downloaded_tiles = {}
        
        for tile in TILES:
            print(f"\n=== Attempting download for tile {tile['id']} ===")
            print(f"Path: {tile['path']}")
            
            tile_files = {}
            success = True
            
            # Try downloading all bands for this tile
            for band in BANDS:
                try:
                    output_file = data_dir / f"{tile['id']}_{band}_10m.jp2"
                    key = f"{tile['path']}/{band}_10m.jp2"
                    
                    print(f"\nDownloading {band}...")
                    # Download file with standard AWS request format
                    s3.download_file(
                        Bucket=BUCKET,
                        Key=key,
                        Filename=str(output_file),
                        ExtraArgs={'RequestPayer': 'requester'}
                    )
                    tile_files[band] = output_file
                    print(f"✓ Successfully downloaded {band}")
                    
                except Exception as e:
                    print(f"Failed to download {band}: {str(e)}")
                    success = False
                    break
            
            if success:
                downloaded_tiles[tile['id']] = tile_files
                print(f"\n✓ Successfully downloaded all bands for tile {tile['id']}")
                break  # Stop if we get a complete tile
            else:
                print(f"\n✗ Failed to download tile {tile['id']}, trying next tile...")
        
        if downloaded_tiles:
            print("\nDownload Summary:")
            for tile_id, files in downloaded_tiles.items():
                print(f"\nTile: {tile_id}")
                for band, path in files.items():
                    print(f"- {band}: {path}")
            return 0
        else:
            print("\nFailed to download any complete tiles")
            return 1
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(download_mangalore_scene())