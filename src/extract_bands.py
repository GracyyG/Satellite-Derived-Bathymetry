"""
Sentinel-2 Band Extraction Module

This module handles extracting specific bands from Sentinel-2 SAFE files
and preparing them for bathymetry analysis.
"""

import os
import zipfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
import rasterio
import numpy as np

logger = logging.getLogger(__name__)

def extract_bands_from_safe(safe_path: str, output_dir: str, bands: List[str] = None) -> Dict[str, str]:
    """
    Extracts specific Sentinel-2 bands from a .SAFE folder or .zip file and saves them as .jp2 files.
    
    Args:
        safe_path: Path to SAFE file or directory
        output_dir: Directory to save extracted bands
        bands: List of band names to extract (default: ['B02', 'B03', 'B04', 'B08'])
    
    Returns:
        Dictionary mapping band names to extracted file paths
    
    Steps:
    1. Unzip the SAFE archive if it's zipped
    2. Locate the IMG_DATA directory under GRANULE
    3. Find matching band filenames (e.g., *_B02_10m.jp2)
    4. Copy or extract them to output_dir, renaming to B02.jp2, B03.jp2, etc.
    5. Return a dict of band name → file path
    """
    if bands is None:
        bands = ['B02', 'B03', 'B04', 'B08']
    
    safe_path = Path(safe_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting bands {bands} from {safe_path}")
    
    # Handle zipped SAFE files
    temp_dir = None
    if safe_path.suffix.lower() == '.zip':
        logger.info("Extracting SAFE zip file...")
        temp_dir = output_dir / 'temp_safe_extract'
        temp_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(safe_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the .SAFE directory inside
        safe_dirs = list(temp_dir.glob('*.SAFE'))
        if not safe_dirs:
            raise FileNotFoundError(f"No .SAFE directory found in {safe_path}")
        safe_dir = safe_dirs[0]
    else:
        safe_dir = safe_path
    
    if not safe_dir.exists():
        raise FileNotFoundError(f"SAFE directory not found: {safe_dir}")
    
    # Find the GRANULE directory
    granule_dir = safe_dir / 'GRANULE'
    if not granule_dir.exists():
        raise FileNotFoundError(f"GRANULE directory not found in {safe_dir}")
    
    # Find the first granule subdirectory
    granule_subdirs = [d for d in granule_dir.iterdir() if d.is_dir()]
    if not granule_subdirs:
        raise FileNotFoundError(f"No granule subdirectories found in {granule_dir}")
    
    granule_subdir = granule_subdirs[0]
    img_data_dir = granule_subdir / 'IMG_DATA'
    
    # Check for different resolution subdirectories (newer SAFE format)
    r10m_dir = img_data_dir / 'R10m'
    r20m_dir = img_data_dir / 'R20m'
    
    extracted_bands = {}
    
    for band in bands:
        logger.info(f"Processing band {band}...")
        
        # Search for band files in different locations
        search_patterns = [
            f"*_{band}_10m.jp2",  # 10m resolution
            f"*_{band}_20m.jp2",  # 20m resolution
            f"*_{band}.jp2",      # Generic
        ]
        
        search_dirs = [img_data_dir, r10m_dir, r20m_dir]
        
        band_file = None
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for pattern in search_patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    band_file = matches[0]
                    break
            
            if band_file:
                break
        
        if not band_file:
            logger.warning(f"Band {band} not found in {safe_dir}")
            continue
        
        # Copy band file to output directory with simplified name
        output_file = output_dir / f"{band}.jp2"
        shutil.copy2(band_file, output_file)
        
        extracted_bands[band] = str(output_file)
        logger.info(f"✅ Extracted {band}: {output_file}")
    
    # Clean up temporary directory
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary extraction directory")
    
    logger.info(f"Successfully extracted {len(extracted_bands)} bands to {output_dir}")
    return extracted_bands


def load_band_as_array(band_path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Load a band file as a numpy array.
    
    Args:
        band_path: Path to the band .jp2 file
        dtype: Output data type
        
    Returns:
        2D numpy array of band values
    """
    with rasterio.open(band_path) as src:
        band_data = src.read(1).astype(dtype)
        
        # Handle nodata values
        if src.nodata is not None:
            band_data[band_data == src.nodata] = np.nan
        
        return band_data


def calculate_water_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate water-related indices from Sentinel-2 bands.
    
    Args:
        bands: Dictionary of band_name -> array
        
    Returns:
        Dictionary of index_name -> array
    """
    indices = {}
    
    # Normalized Difference Water Index
    if 'B03' in bands and 'B08' in bands:
        b03, b08 = bands['B03'], bands['B08']
        indices['NDWI'] = (b03 - b08) / (b03 + b08 + 1e-8)
    
    # Modified Normalized Difference Water Index  
    if 'B03' in bands and 'B11' in bands:
        b03, b11 = bands['B03'], bands['B11']
        indices['MNDWI'] = (b03 - b11) / (b03 + b11 + 1e-8)
    elif 'B03' in bands and 'B02' in bands:
        # Fallback using B02 instead of B11
        b03, b02 = bands['B03'], bands['B02']
        indices['MNDWI'] = (b03 - b02) / (b03 + b02 + 1e-8)
    
    # Blue-Red ratio (simple depth indicator)
    if 'B02' in bands and 'B04' in bands:
        b02, b04 = bands['B02'], bands['B04']
        indices['BR_ratio'] = b02 / (b04 + 1e-6)
    
    # Green-Red ratio
    if 'B03' in bands and 'B04' in bands:
        b03, b04 = bands['B03'], bands['B04']
        indices['GR_ratio'] = b03 / (b04 + 1e-6)
    
    logger.info(f"Calculated {len(indices)} water indices: {list(indices.keys())}")
    return indices


def create_feature_stack(bands: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Stack bands and indices into a feature array.
    
    Args:
        bands: Dictionary of band arrays
        indices: Dictionary of index arrays
        
    Returns:
        3D array of shape (height, width, n_features)
    """
    all_arrays = {**bands, **indices}
    
    if not all_arrays:
        raise ValueError("No bands or indices provided")
    
    # Get common shape
    first_array = next(iter(all_arrays.values()))
    height, width = first_array.shape
    
    # Stack all features
    feature_list = []
    feature_names = []
    
    for name, array in all_arrays.items():
        if array.shape == (height, width):
            feature_list.append(array)
            feature_names.append(name)
        else:
            logger.warning(f"Skipping {name} due to shape mismatch: {array.shape} vs {(height, width)}")
    
    if not feature_list:
        raise ValueError("No compatible arrays found for stacking")
    
    features = np.stack(feature_list, axis=-1)
    logger.info(f"Created feature stack with shape {features.shape} using: {feature_names}")
    
    return features, feature_names