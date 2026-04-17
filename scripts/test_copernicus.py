"""
Test script for Copernicus API connection and basic search functionality.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project source to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from download_sentinel import authenticate_copernicus, search_sentinel_scenes
from shapely.geometry import box
import geopandas as gpd

def create_test_aoi():
    """Create a test AOI around a coastal area."""
    # Example coordinates for a coastal area (adjust these for your area of interest)
    bbox = box(-123.0, 45.0, -122.0, 46.0)  # Example: Oregon coast
    gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326')
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Save as GeoJSON
    aoi_path = data_dir / 'aoi.geojson'
    gdf.to_file(aoi_path, driver='GeoJSON')
    return aoi_path

def main():
    # Get credentials
    username = input("Enter your Copernicus username: ")
    password = input("Enter your Copernicus password: ")
    
    try:
        # Test authentication
        print("\nTesting authentication...")
        api = authenticate_copernicus(username, password)
        print("✓ Authentication successful!")
        
        # Create test AOI
        print("\nCreating test Area of Interest...")
        aoi_path = create_test_aoi()
        print(f"✓ AOI created at {aoi_path}")
        
        # Search for scenes
        print("\nSearching for Sentinel-2 scenes...")
        date_range = (
            datetime.now() - timedelta(days=30),  # Last 30 days
            datetime.now()
        )
        
        scenes = search_sentinel_scenes(
            api=api,
            aoi_path=aoi_path,
            date_range=date_range,
            max_cloud_cover=20.0
        )
        
        # Display results
        print(f"\nFound {len(scenes)} scenes:")
        for scene in scenes:
            print(f"\nScene ID: {scene['id']}")
            print(f"Title: {scene['title']}")
            print(f"Date: {scene['date']}")
            print(f"Cloud Cover: {scene['cloud_cover']}%")
            print(f"Size: {scene['size']}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())