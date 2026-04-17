"""
Access Sentinel-2 data from AWS using product information
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSS2Access:
    """Handle AWS Sentinel-2 data access"""
    
    def __init__(self):
        # Configure anonymous S3 access
        self.s3 = boto3.client(
            's3',
            config=Config(
                signature_version=UNSIGNED,
                retries={'max_attempts': 3},
                connect_timeout=10,
                read_timeout=30
            )
        )
        
        self.bucket = "sentinel-s2-l2a"
        self.required_bands = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
        
    def _parse_product_id(self, product_title: str) -> Dict[str, str]:
        """Parse Sentinel-2 product title to get components"""
        
        # Example: S2B_MSIL2A_20231015T051649_N0509_R062_T43PGQ_20231015T070630
        pattern = r"S2[AB]_MSIL2A_(\d{8}T\d{6})_.*_T(\d{2}[A-Z]{3})_"
        match = re.search(pattern, product_title)
        
        if not match:
            return {}
            
        sensing_time = match.group(1)  # YYYYMMDDTHHMMSS
        tile_id = match.group(2)       # e.g., 43PGQ
        
        # Extract date components
        year = sensing_time[0:4]
        month = sensing_time[4:6].lstrip('0')  # Remove leading zero
        day = sensing_time[6:8].lstrip('0')    # Remove leading zero
        
        return {
            'tile_id': tile_id,
            'year': year,
            'month': month,
            'day': day,
            'sensing_time': sensing_time
        }
    
    def get_aws_path(self, product_title: str) -> Optional[str]:
        """Convert Copernicus product title to AWS path"""
        
        info = self._parse_product_id(product_title)
        if not info:
            return None
            
        # AWS path format: tiles/43/P/GQ/2023/10/15
        tile = info['tile_id']
        path = f"tiles/{tile[0:2]}/{tile[2]}/{tile[3:5]}/{info['year']}/{info['month']}/{info['day']}"
        
        return path
    
    def download_bands(self, product_title: str, output_dir: Path) -> bool:
        """Download required bands for a product"""
        
        aws_path = self.get_aws_path(product_title)
        if not aws_path:
            logger.error(f"Could not parse product title: {product_title}")
            return False
            
        info = self._parse_product_id(product_title)
        if not info:
            return False
            
        tile_id = info['tile_id']
        success = True
        downloaded = {}
        
        print(f"\n=== Downloading bands for {tile_id} ===")
        print(f"Date: {info['year']}-{info['month']}-{info['day']}")
        print(f"AWS Path: {aws_path}")
        
        for band in self.required_bands:
            try:
                output_file = output_dir / f"{tile_id}_{band}_10m.jp2"
                key = f"{aws_path}/{band}_10m.jp2"
                
                print(f"\nDownloading {band}...")
                self.s3.download_file(
                    Bucket=self.bucket,
                    Key=key,
                    Filename=str(output_file)
                )
                downloaded[band] = output_file
                print(f"✓ Successfully downloaded {band}")
                
            except Exception as e:
                print(f"Failed to download {band}: {str(e)}")
                success = False
                break
        
        if success:
            print(f"\n✓ Successfully downloaded all bands for {tile_id}")
            return True
        else:
            print(f"\n✗ Failed to download complete set for {tile_id}")
            return False

def main():
    """Main function to test AWS access"""
    
    # Read cached scene information
    cache_dir = Path("data/cache")
    data_dir = Path("data/sentinel/mangalore")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find most recent cache file
    cache_files = list(cache_dir.glob("scenes_*.json"))
    if not cache_files:
        print("No cached scene information found. Run find_sentinel_scenes.py first.")
        return 1
    
    latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_cache) as f:
        scenes = json.load(f)
    
    if not scenes:
        print("No scenes found in cache")
        return 1
    
    # Try downloading the most recent scene
    aws = AWSS2Access()
    scene = scenes[0]  # Most recent scene
    
    success = aws.download_bands(scene['title'], data_dir)
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())