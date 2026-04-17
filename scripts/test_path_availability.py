"""
Simple test script to check Sentinel-2 data availability for Mangalore
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_path_availability():
    """Test different paths in the Sentinel-2 bucket"""
    
    s3 = boto3.client(
        's3',
        config=Config(
            signature_version=UNSIGNED,
            retries={'max_attempts': 3},
            connect_timeout=10,
            read_timeout=30
        )
    )
    
    bucket = "sentinel-s2-l2a"
    
    # Test a known recent date and location
    test_paths = [
        "tiles/43/P/GQ/2023",  # Mangalore area
        "products/2023",       # Alternative path structure
        "tiles/43/P"          # Parent directory
    ]
    
    print("\nTesting AWS Sentinel-2 data access...")
    
    for path in test_paths:
        try:
            print(f"\nChecking path: {path}")
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=path,
                MaxKeys=5
            )
            
            if 'Contents' in response:
                print("Found files:")
                for obj in response['Contents']:
                    print(f"- {obj['Key']}")
            else:
                print("No files found")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(test_path_availability())