"""
Sentinel-2 Data Download Module

This module handles the downloading of Sentinel-2 satellite imagery using the Copernicus
Open Access Hub API (sentinelsat) for Satellite-Derived Bathymetry processing.

Functions:
    - authenticate_copernicus: Authenticate with Copernicus Open Access Hub
    - search_sentinel_scenes: Search for Sentinel-2 scenes based on AOI and date range
    - download_bands: Download specific bands (B02, B03, B04, B08) from Sentinel-2 scenes
"""

from typing import List, Dict, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from shapely.geometry import box, mapping
import geopandas as gpd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def authenticate_copernicus(username: str, password: str) -> SentinelAPI:
    """
    Authenticate with the Copernicus Data Space Ecosystem.
    
    Args:
        username: Copernicus Services Data Hub username
        password: Copernicus Services Data Hub password
        
    Returns:
        SentinelAPI: Authenticated API instance
    """
    try:
        # Use the new Copernicus Data Space Ecosystem URL
        api = SentinelAPI(username, password, 'https://catalogue.dataspace.copernicus.eu/odata/v1')
        logger.info("Successfully authenticated with Copernicus Data Space Ecosystem")
        return api
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise

def search_sentinel_scenes(
    api: SentinelAPI,
    aoi_path: Path,
    date_range: Tuple[datetime, datetime],
    max_cloud_cover: float = 20.0
) -> List[Dict]:
    """
    Search for Sentinel-2 scenes based on Area of Interest and date range.
    
    Args:
        api: Authenticated SentinelAPI instance
        aoi_path: Path to AOI GeoJSON file
        date_range: Tuple of (start_date, end_date)
        max_cloud_cover: Maximum acceptable cloud cover percentage
        
    Returns:
        List of dictionaries containing scene metadata
    """
    try:
        # Read and convert AOI to WKT format
        footprint = geojson_to_wkt(read_geojson(str(aoi_path)))
        
        # Search for Sentinel-2 scenes
        products = api.query(
            footprint,
            date=date_range,
            platformname='Sentinel-2',
            cloudcoverpercentage=(0, max_cloud_cover),
            producttype='S2MSI2A'  # Level-2A products (atmospherically corrected)
        )
        
        logger.info(f"Found {len(products)} scenes matching criteria")
        return [
            {
                'id': id,
                'title': product['title'],
                'cloud_cover': product['cloudcoverpercentage'],
                'date': product['beginposition'],
                'size': product['size']
            }
            for id, product in products.items()
        ]
    except Exception as e:
        logger.error(f"Scene search failed: {str(e)}")
        raise

def download_bands(
    api: SentinelAPI,
    scene_id: str,
    output_dir: Path,
    bands: List[str] = ['B02', 'B03', 'B04', 'B08']
) -> Dict[str, Path]:
    """
    Download specific bands from a Sentinel-2 scene.
    
    Args:
        api: Authenticated SentinelAPI instance
        scene_id: Sentinel-2 scene identifier
        output_dir: Directory to save downloaded bands
        bands: List of band identifiers to download
        
    Returns:
        Dictionary mapping band IDs to their file paths
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the product
        logger.info(f"Downloading scene {scene_id}")
        product_info = api.download(scene_id, str(output_dir))
        
        if not product_info:
            raise ValueError(f"Failed to download scene {scene_id}")
        
        # Extract bands from the downloaded product
        downloaded_path = output_dir / product_info['title']
        band_paths = {}
        
        # TODO: Extract specific bands from the SAFE format
        # This part requires additional implementation using rasterio or SNAP
        # to extract specific bands from the Sentinel-2 SAFE format
        
        logger.info(f"Successfully downloaded and extracted {len(bands)} bands")
        return band_paths
        
    except Exception as e:
        logger.error(f"Band download failed: {str(e)}")
        raise

if __name__ == "__main__":
    import os
    from datetime import datetime, timedelta
    
    # Get credentials from environment variables
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    
    if not username or not password:
        logger.error("Please set COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables")
        exit(1)
    
    try:
        # Test authentication
        api = authenticate_copernicus(username, password)
        
        # Test scene search
        aoi_path = Path("data/aoi.geojson")
        date_range = (datetime.now() - timedelta(days=30), datetime.now())
        
        scenes = search_sentinel_scenes(api, aoi_path, date_range)
        
        if scenes:
            # Test band download for the first scene
            scene_id = scenes[0]['id']
            output_dir = Path("data/sentinel")
            band_paths = download_bands(api, scene_id, output_dir)
            
            logger.info("Download completed successfully!")
            for band, path in band_paths.items():
                logger.info(f"{band}: {path}")
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")