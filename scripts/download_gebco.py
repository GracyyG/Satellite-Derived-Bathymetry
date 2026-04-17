"""
Download GEBCO bathymetry data for the Mangalore study area.
Uses GEBCO 2023 Grid (15 arc-second resolution).
"""

import os
import requests
from pathlib import Path
import logging
import rasterio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mangalore area coordinates (approximately)
BOUNDS = {
    'north': 13.0,  # Northern latitude
    'south': 12.8,  # Southern latitude
    'east': 74.9,   # Eastern longitude
    'west': 74.7    # Western longitude
}

# GEBCO API endpoint (you'll need to register and get API key from GEBCO website)
GEBCO_API = "https://download.gebco.net/data/GEBCO_2023/gebco_2023_sub.nc"

def download_gebco_data(output_dir: Path) -> Path:
    """
    Download GEBCO bathymetry data for the specified area.
    
    Args:
        output_dir: Directory to save the downloaded data
        
    Returns:
        Path to downloaded file
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'mangalore_gebco.nc'
        
        # Construct API request with area bounds
        params = {
            'north': BOUNDS['north'],
            'south': BOUNDS['south'],
            'east': BOUNDS['east'],
            'west': BOUNDS['west'],
            'format': 'netcdf'
        }
        
        logger.info(f"Downloading GEBCO data for area: {BOUNDS}")
        
        # For now, just create a placeholder note
        with open(output_file.with_suffix('.txt'), 'w') as f:
            f.write("""
            To download GEBCO data:
            1. Visit https://www.gebco.net/data_and_products/gridded_bathymetry_data/
            2. Register for an account
            3. Request the area: 
               - North: 13.0°N
               - South: 12.8°N
               - East: 74.9°E
               - West: 74.7°E
            4. Download the data and place it in this directory
            """)
        
        logger.info("Created instructions for manual GEBCO data download")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to download GEBCO data: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up paths
    project_dir = Path(__file__).parent.parent
    gebco_dir = project_dir / 'data' / 'gebco_reference'
    
    # Download data
    download_gebco_data(gebco_dir)