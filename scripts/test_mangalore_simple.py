"""
Download Sentinel-2 data for New Mangalore Port (2025-11-01)
Coordinates: 12.92°N, 74.82°E
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_available_scenes():
    session = boto3.Session(region_name='eu-central-1')
    s3 = session.client(
        's3',
        config=Config(signature_version=UNSIGNED)
    )

    # First verify we can access AWS and list buckets
    try:
        # Check access to the Sentinel bucket
        print("Checking AWS connection...")
        response = s3.list_buckets()
        print("\nAvailable buckets:")
        for bucket in response['Buckets']:
            print(f"- {bucket['Name']}")
    except Exception as e:
        print(f"Error accessing AWS: {str(e)}")
        return None

    # Search parameters for Mangalore
    BUCKET = "sentinel-s2-l2a"
    PREFIX = "tiles/43/P/GP/"  # Remove year to find any available data

    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=BUCKET,
            Prefix=PREFIX,
            PaginationConfig={'MaxItems': 10}
        )
        
        print("\nAvailable scenes for Mangalore (Tile 43PGP):")
        scene_dates = []
        
        for page in pages:
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('B02_10m.jp2'):
                    scene_date = obj['Key'].split('/')[4:7]
                    scene_dates.append('/'.join(scene_date))
                    print(f"Found scene from: {'/'.join(scene_date)}")
        
        if scene_dates:
            return scene_dates[0]  # Return most recent scene date
        return None
            
    except Exception as e:
        print(f"Error listing scenes: {str(e)}")
        return None

def download_mangalore_scene():
    try:
        # Create directories
        data_dir = Path("data/sentinel/mangalore")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Find available scenes first
        latest_scene = find_available_scenes()
        if not latest_scene:
            print("No available scenes found")
            return 1
        
        # Configure anonymous S3 access
        s3 = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Mangalore scene parameters
        TILE_ID = "43PGP"
        SCENE_PATH = f"tiles/43/P/GP/{latest_scene}"
        BUCKET = "sentinel-s2-l2a"
        
        print("\n=== Downloading Mangalore Port Scene ===")
        print(f"Date: {latest_scene}")
        print("Tile ID:", TILE_ID)
        print("Location: 12.95°N, 74.80°E")
        
        # Download required bands
        bands = ['B02', 'B03', 'B04', 'B08']
        band_files = {}
        
        for band in bands:
            try:
                output_file = data_dir / f"{band}_10m.jp2"
                key = f"{SCENE_PATH}/{band}_10m.jp2"
                
                print(f"\nDownloading {band}...")
                s3.download_file(
                    Bucket=BUCKET,
                    Key=key,
                    Filename=str(output_file),
                    ExtraArgs={'ExpectedBucketOwner': 'sentinel-s2-l2a-aws'}
                )
                band_files[band] = output_file
                print(f"✓ Successfully downloaded {band}")
                
            except Exception as e:
                print(f"Failed to download {band}: {str(e)}")
        
        print("\nDownload Summary:")
        for band, path in band_files.items():
            print(f"Band {band}: {path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(download_mangalore_scene())