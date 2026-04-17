"""
Download Sentinel-2 data for Mangalore using verified AWS path structure
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_mangalore_data():
    """Download Sentinel-2 data for Mangalore"""
    
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
        
        # Verified AWS path structure
        bucket = "sentinel-s2-l2a"
        tile_id = "43PGQ"  # Mangalore tile
        base_path = f"tiles/43/P/GQ/2023/1/1/0/R10m"  # Using verified path
        bands = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
        
        print("\nDownloading Sentinel-2 data for Mangalore...")
        print("Tile ID:", tile_id)
        print("Path:", base_path)
        
        downloaded_files = {}
        
        for band in bands:
            try:
                output_file = data_dir / f"{tile_id}_{band}.jp2"
                key = f"{base_path}/{band}.jp2"
                
                print(f"\nDownloading {band}...")
                s3.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=str(output_file)
                )
                downloaded_files[band] = output_file
                print(f"✓ Successfully downloaded {band}")
                
            except Exception as e:
                print(f"Failed to download {band}: {str(e)}")
                return 1
        
        # Print summary
        if downloaded_files:
            print("\nDownload Summary:")
            for band, path in downloaded_files.items():
                print(f"Band {band}: {path}")
            return 0
        else:
            print("\nNo files were downloaded successfully")
            return 1
            
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(download_mangalore_data())