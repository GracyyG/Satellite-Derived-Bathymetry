#!/usr/bin/env python3
"""
Sentinel-2 Data Downloader

Downloads Sentinel-2 Level-2A data for a specified region using the Copernicus Data Space.
Handles authentication, product search, and selective band download.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import requests
from sentinelsat import SentinelAPI, geojson_to_wkt
from shapely.geometry import box
import geopandas as gpd

class Sentinel2Downloader:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_path = project_root / 'config' / 'location_config.json'
        self.auth_path = project_root / 'config' / 'copernicus_auth.json'
        
        # Set up logging
        self.log_dir = project_root / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Load configuration
        self.load_config()
        
        # Set up paths
        self.region_slug = self.config['region_name'].lower().replace(' ', '_')
        self.data_dir = project_root / 'data' / 'sentinel' / self.region_slug / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Configure logging to file and console"""
        log_file = self.log_dir / 'sentinel_download_log.txt'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """Load region configuration"""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
            self.logger.info(f"Loaded configuration for region: {self.config['region_name']}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file: {self.config_path}")
            sys.exit(1)
            
    def get_auth_credentials(self) -> Dict[str, str]:
        """Get Copernicus Data Space credentials"""
        if not self.auth_path.exists():
            print("\n⚠️  Copernicus authentication required!")
            print("Please enter your Copernicus Data Space credentials:")
            username = input("Username: ")
            password = input("Password: ")
            
            auth_data = {
                'username': username,
                'password': password
            }
            
            with open(self.auth_path, 'w') as f:
                json.dump(auth_data, f)
            
            print("Credentials saved to config/copernicus_auth.json")
        
        with open(self.auth_path) as f:
            return json.load(f)
            
    def create_search_bbox(self) -> str:
        """Create WKT representation of search area"""
        bbox = box(
            self.config['aoi']['min_lon'],
            self.config['aoi']['min_lat'],
            self.config['aoi']['max_lon'],
            self.config['aoi']['max_lat']
        )
        return geojson_to_wkt(gpd.GeoSeries([bbox]).__geo_interface__)
        
    def check_existing_data(self) -> bool:
        """Check if required bands already exist"""
        required_bands = ['B02', 'B03', 'B04', 'B08']
        for band in required_bands:
            if not list(self.data_dir.glob(f'*_{band}.jp2')):
                return False
        return True
        
    def download_sentinel_data(self):
        """Download Sentinel-2 L2A data for the specified region"""
        if self.check_existing_data():
            print(f"✅ Sentinel-2 data already present for {self.config['region_name']}")
            return True
            
        # Get authentication credentials
        auth = self.get_auth_credentials()
        
        # Initialize API
        api = SentinelAPI(
            auth['username'],
            auth['password'],
            'https://catalogue.dataspace.copernicus.eu/api/hub/'
        )
        
        # Set search parameters
        footprint = geojson_to_wkt(self.create_search_bbox())
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # Last 6 months
        
        self.logger.info("Searching for Sentinel-2 products...")
        print(f"\n🔍 Searching for Sentinel-2 data over {self.config['region_name']}...")
        
        try:
            # Query products
            products = api.query(
                footprint,
                date=(start_date, end_date),
                platformname='Sentinel-2',
                processinglevel='Level-2A',
                cloudcoverpercentage=(0, 20)
            )
            
            if not products:
                self.logger.error("No suitable products found!")
                return False
                
            # Sort by cloud cover and get best product
            products_df = api.to_dataframe(products)
            best_product = products_df.sort_values('cloudcoverpercentage').iloc[0]
            
            # Log product details
            self.logger.info(f"Selected product: {best_product['title']}")
            self.logger.info(f"Cloud cover: {best_product['cloudcoverpercentage']:.1f}%")
            self.logger.info(f"Date: {best_product['beginposition'].strftime('%Y-%m-%d')}")
            
            # Download product
            print(f"\n📥 Downloading product: {best_product['title']}")
            api.download(best_product['uuid'], directory_path=str(self.data_dir))
            
            print(f"✅ Sentinel-2 data download completed for {self.config['region_name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            return False

def main():
    """Main entry point"""
    try:
        # Get project root
        project_root = Path(__file__).parent
        
        # Initialize downloader
        downloader = Sentinel2Downloader(project_root)
        
        # Print header
        print("\n" + "="*60)
        print(f"🛰️  Downloading Sentinel-2 data for region: {downloader.config['region_name']}")
        print("Area of Interest:")
        print(f"  Latitude:  {downloader.config['aoi']['min_lat']}° to {downloader.config['aoi']['max_lat']}°")
        print(f"  Longitude: {downloader.config['aoi']['min_lon']}° to {downloader.config['aoi']['max_lon']}°")
        print("="*60 + "\n")
        
        # Run download
        success = downloader.download_sentinel_data()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()