"""
Search for available Sentinel-2 scenes over Mangalore using Copernicus API
"""

from datetime import datetime, timedelta
import requests
import os
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CopernicusAPI:
    """Handle Copernicus Open Access Hub API interactions"""
    
    def __init__(self):
        # Use alternative API endpoint
        self.auth_url = "https://apihub.copernicus.eu/apihub"
        self.search_url = f"{self.auth_url}/search"
        
        # Set timeout values
        self.timeout = (30, 60)  # (connect timeout, read timeout)
        
        # Mangalore coordinates
        self.lat = 12.92
        self.lon = 74.82
        
        # Create cache directory
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def search_scenes(self, start_date: str, end_date: str, max_cloud: int = 20) -> List[Dict]:
        """Search for Sentinel-2 scenes"""
        
        try:
            # Build search query
            query = (
                f'platformname:Sentinel-2 AND '
                f'cloudcoverpercentage:[0 TO {max_cloud}] AND '
                f'footprint:"Intersects(POINT({self.lon} {self.lat}))" AND '
                f'beginposition:[{start_date}T00:00:00.000Z TO {end_date}T23:59:59.999Z] AND '
                f'producttype:S2MSI2A'  # L2A products
            )
            
            params = {
                'q': query,
                'rows': 100,
                'orderby': 'beginposition desc'
            }
            
            print(f"\nSearching for scenes between {start_date} and {end_date}")
            print(f"Location: {self.lat}°N, {self.lon}°E")
            print(f"Max cloud cover: {max_cloud}%")
            
            response = requests.get(
                self.search_url,
                params=params,
                timeout=self.timeout,
                verify=False  # Skip SSL verification for now
            )
            
            if response.status_code == 401:
                print("Error: Authentication required. Please provide credentials.")
                return []
            elif response.status_code != 200:
                print(f"Error: API returned status code {response.status_code}")
                return []
                
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract scene information
            scenes = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                try:
                    scene = {
                        'title': entry.find('.//{http://www.w3.org/2005/Atom}title').text,
                        'id': entry.find('.//{http://www.w3.org/2005/Atom}id').text,
                        'link': entry.find('.//{http://www.w3.org/2005/Atom}link').attrib['href'],
                        'date': entry.find('.//{http://www.w3.org/2005/Atom}date').text,
                        'footprint': entry.find('.//{http://www.w3.org/2005/Atom}str[@name="footprint"]').text,
                        'cloud_cover': float(entry.find('.//{http://www.w3.org/2005/Atom}double[@name="cloudcoverpercentage"]').text)
                    }
                    scenes.append(scene)
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Could not parse scene: {str(e)}")
                    continue
            
            # Save results to cache
            cache_file = self.cache_dir / f"scenes_{start_date}_{end_date}.json"
            with open(cache_file, 'w') as f:
                json.dump(scenes, f, indent=2)
            
            # Print summary
            print(f"\nFound {len(scenes)} scenes:")
            for scene in scenes:
                print(f"\nScene: {scene['title']}")
                print(f"Date: {scene['date']}")
                print(f"Cloud Cover: {scene['cloud_cover']}%")
            
            return scenes
            
        except Exception as e:
            logger.error(f"Error searching scenes: {str(e)}")
            return []

def main():
    """Main function to search for scenes"""
    api = CopernicusAPI()
    
    # Search for scenes in last 6 months
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    scenes = api.search_scenes(start_date, end_date)
    
    if not scenes:
        print("No scenes found")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())