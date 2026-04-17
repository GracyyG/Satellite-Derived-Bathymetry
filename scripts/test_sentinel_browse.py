"""
Script to browse available Sentinel-2 scenes
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def browse_sentinel_scenes():
    """Browse available Sentinel-2 scenes"""
    try:
        print("Configuring AWS anonymous access...")
        session = boto3.Session()
        s3 = session.client('s3', config=Config(
            signature_version=UNSIGNED,
            connect_timeout=5,
            read_timeout=5,
            retries={'max_attempts': 2}
        ))
        
        print("\nVerifying Sentinel-2 bucket access...")
        try:
            s3.head_bucket(Bucket="sentinel-s2-l2a")
            print("✓ Successfully connected to Sentinel-2 bucket")
        except Exception as e:
            print(f"Error: Cannot access Sentinel-2 bucket: {str(e)}")
            return 1
        
        print("\nSearching for New Mangalore Port scenes...")
        # Coordinates from Copernicus Browser: ~12.92°N, 74.82°E
        # Date: 2025-11-01
        
        # The 100km grid squares to check
        grid_squares = ['GP', 'GN', 'GQ', 'FP', 'FN']
        
        # Build all possible paths
        test_paths = []
        for square in grid_squares:
            # Try both with and without leading zero in day
            test_paths.extend([
                f"tiles/43/P/{square}/2025/11/01",  # With leading zero
                f"tiles/43/P/{square}/2025/11/1",   # Without leading zero
            ])
            
        print("Checking the following possible paths:")
        for path in test_paths:
            print(f"- {path}")
        
        print("Searching for available Sentinel-2 scenes...")
        
        for path in test_paths:
            print(f"\nChecking path: {path}")
            try:
                # Use paginator for more reliable results
                paginator = s3.get_paginator('list_objects_v2')
                operation_parameters = {
                    'Bucket': "sentinel-s2-l2a",
                    'Prefix': path + "/",
                    'PaginationConfig': {
                        'MaxItems': 10
                    }
                }
                
                page_iterator = paginator.paginate(**operation_parameters)
                
                found_files = False
                for page in page_iterator:
                    if 'Contents' in page:
                        found_files = True
                        print("Found files:")
                        for obj in page['Contents']:
                            key = obj['Key']
                            if 'B02_10m.jp2' in key:
                                parts = key.split('/')
                                tile_id = f"{parts[1]}{parts[2]}{parts[3]}"
                                date = f"{parts[4]}-{parts[5]}-{parts[6]}"
                                size_mb = obj['Size'] / (1024 * 1024)
                                print(f"- Tile: {tile_id}, Date: {date}, Size: {size_mb:.1f}MB")
                                print(f"  Path: {key}")
                
                if not found_files:
                    print("No files found in this path")
                    
            except Exception as e:
                print(f"Error checking path {path}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Failed to access Sentinel-2 data: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(browse_sentinel_scenes())