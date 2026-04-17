"""
Image Preprocessing Module

This module handles the preprocessing of Sentinel-2 imagery for SDB analysis,
including atmospheric correction, resampling, and feature extraction.

Functions:
    - correct_atmosphere: Apply atmospheric correction to Sentinel-2 bands
    - calculate_indices: Calculate water indices (NDWI, etc.)
    - mask_water: Create water mask from indices
    - extract_features: Extract features for SDB model training
"""

import numpy as np
from typing import List, Dict, Union, Tuple
from pathlib import Path
import rasterio
import logging
from rasterio.warp import reproject, Resampling
import cv2
import xarray as xr
from rasterio.mask import mask
import rioxarray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mask_clouds(
    qa60_path: Path,
    data_mask: np.ndarray = None
) -> np.ndarray:
    """
    Create cloud mask from Sentinel-2 QA60 band.
    
    Args:
        qa60_path: Path to QA60 band file
        data_mask: Optional existing mask to combine with cloud mask
        
    Returns:
        Boolean mask array where True indicates clear (non-cloud) pixels
    """
    try:
        with rasterio.open(qa60_path) as src:
            qa_data = src.read(1)
            
            # QA60 band bit information
            # Bit 10: Opaque clouds
            # Bit 11: Cirrus clouds
            cloud_mask = np.bitwise_and(qa_data, 1 << 10) == 0  # Clear of opaque clouds
            cirrus_mask = np.bitwise_and(qa_data, 1 << 11) == 0  # Clear of cirrus clouds
            
            # Combine masks
            clear_mask = cloud_mask & cirrus_mask
            
            if data_mask is not None:
                clear_mask = clear_mask & data_mask
            
            logger.info(f"Created cloud mask with {np.sum(clear_mask)} clear pixels")
            return clear_mask
            
    except Exception as e:
        logger.error(f"Cloud masking failed: {str(e)}")
        raise

def correct_atmosphere(
    band_paths: Dict[str, Path]
) -> Dict[str, np.ndarray]:
    """
    Apply atmospheric correction to Sentinel-2 bands.
    Since we're using Level-2A products, they're already atmospherically corrected,
    but we'll normalize the values and handle any remaining artifacts.
    
    Args:
        band_paths: Dictionary mapping band IDs to file paths
        
    Returns:
        Dictionary of atmospherically corrected bands as NumPy arrays
    """
    corrected_bands = {}
    reference_profile = None
    
    try:
        # Read and normalize bands
        for band_name, band_path in band_paths.items():
            with rasterio.open(band_path) as src:
                # Store reference profile from first band
                if reference_profile is None:
                    reference_profile = src.profile
                
                # Read band data
                band_data = src.read(1).astype(np.float32)
                
                # Apply basic normalization (0-1 range)
                band_min = np.percentile(band_data, 2)  # Remove outliers
                band_max = np.percentile(band_data, 98)
                band_normalized = np.clip((band_data - band_min) / (band_max - band_min), 0, 1)
                
                # Apply simple dehazing if needed
                if band_name in ['B02', 'B03', 'B04']:  # Visible bands
                    dark_channel = cv2.erode(band_normalized, np.ones((3,3)))
                    atmospheric_light = np.max(dark_channel)
                    transmission = 1 - 0.95 * dark_channel / atmospheric_light
                    transmission = cv2.max(transmission, 0.1)
                    band_normalized = (band_normalized - atmospheric_light) / transmission + atmospheric_light
                
                corrected_bands[band_name] = band_normalized
                
        logger.info(f"Successfully corrected {len(corrected_bands)} bands")
        return corrected_bands
        
    except Exception as e:
        logger.error(f"Atmospheric correction failed: {str(e)}")
        raise

def calculate_indices(
    bands: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculate water indices from Sentinel-2 bands.
    
    Args:
        bands: Dictionary of band arrays
        
    Returns:
        Dictionary of calculated indices including:
        - NDWI (Normalized Difference Water Index)
        - MNDWI (Modified NDWI)
        - AWEI (Automated Water Extraction Index)
    """
    indices = {}
    
    try:
        # Normalized Difference Water Index (NDWI)
        # Uses Green and NIR bands
        indices['NDWI'] = (bands['B03'] - bands['B08']) / (bands['B03'] + bands['B08'] + 1e-6)
        
        # Modified NDWI (MNDWI)
        # Here we use Blue instead of SWIR (not available in our band selection)
        indices['MNDWI'] = (bands['B03'] - bands['B02']) / (bands['B03'] + bands['B02'] + 1e-6)
        
        # Simple Ratio (SR)
        indices['SR'] = bands['B03'] / (bands['B04'] + 1e-6)
        
        # Blue/Red ratio (useful for bathymetry)
        indices['BR_ratio'] = bands['B02'] / (bands['B04'] + 1e-6)
        
        logger.info(f"Calculated indices: {list(indices.keys())}")
        return indices
        
    except Exception as e:
        logger.error(f"Index calculation failed: {str(e)}")
        raise

def mask_water(
    indices: Dict[str, np.ndarray],
    threshold: float = 0.0
) -> np.ndarray:
    """
    Create water mask from calculated indices.
    
    Args:
        indices: Dictionary of calculated indices
        threshold: Threshold for water classification
        
    Returns:
        Boolean mask array identifying water pixels
    """
    try:
        # Use NDWI as primary water detector
        water_mask = indices['NDWI'] > threshold
        
        # Refine mask using MNDWI
        water_mask = water_mask & (indices['MNDWI'] > threshold)
        
        # Additional refinement using band ratios
        water_mask = water_mask & (indices['BR_ratio'] > 1.0)  # Water reflects more blue than red
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        water_mask = water_mask.astype(np.uint8)
        
        # Remove noise
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        
        logger.info(f"Created water mask with {np.sum(water_mask)} water pixels")
        return water_mask.astype(bool)
        
    except Exception as e:
        logger.error(f"Water masking failed: {str(e)}")
        raise

def extract_features(
    bands: Dict[str, np.ndarray],
    indices: Dict[str, np.ndarray],
    water_mask: np.ndarray
) -> np.ndarray:
    """
    Extract features for SDB model training.
    
    Args:
        bands: Dictionary of preprocessed band arrays
        indices: Dictionary of calculated indices
        water_mask: Boolean water mask array
        
    Returns:
        Feature array for model training (N x M array where N is number of water pixels
        and M is number of features)
    """
    try:
        # List of features to extract
        band_features = ['B02', 'B03', 'B04', 'B08']  # Raw bands
        index_features = ['NDWI', 'MNDWI', 'SR', 'BR_ratio']  # Calculated indices
        
        # Initialize feature list
        features = []
        
        # Add band values
        for band_name in band_features:
            features.append(bands[band_name][water_mask])
            
        # Add index values
        for index_name in index_features:
            features.append(indices[index_name][water_mask])
            
        # Add band ratios (all combinations)
        for i, b1 in enumerate(band_features):
            for j, b2 in enumerate(band_features[i+1:], i+1):
                ratio = bands[b1][water_mask] / (bands[b2][water_mask] + 1e-6)
                features.append(ratio)
        
        # Stack features into 2D array
        feature_array = np.vstack(features).T
        
        logger.info(f"Extracted features array with shape {feature_array.shape}")
        return feature_array
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise

def create_xarray_dataset(
    bands: Dict[str, np.ndarray],
    indices: Dict[str, np.ndarray],
    water_mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str
) -> xr.Dataset:
    """
    Convert processed data to xarray Dataset with spatial coordinates.
    
    Args:
        bands: Dictionary of preprocessed band arrays
        indices: Dictionary of calculated indices
        water_mask: Boolean water mask array
        transform: Rasterio affine transform
        crs: Coordinate reference system
        
    Returns:
        xarray Dataset with all bands, indices, and masks
    """
    try:
        # Create coordinates
        height, width = next(iter(bands.values())).shape
        x_coords = np.arange(width) * transform[0] + transform[2]
        y_coords = np.arange(height) * transform[4] + transform[5]
        
        # Create dataset
        data_vars = {}
        
        # Add bands
        for name, data in bands.items():
            data_vars[name] = (["y", "x"], data)
            
        # Add indices
        for name, data in indices.items():
            data_vars[name] = (["y", "x"], data)
            
        # Add mask
        data_vars["water_mask"] = (["y", "x"], water_mask.astype(np.int8))
        
        # Create dataset with coordinates
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "x": x_coords,
                "y": y_coords
            }
        )
        
        # Add CRS information
        ds.rio.write_crs(crs, inplace=True)
        
        logger.info(f"Created xarray Dataset with variables: {list(ds.data_vars)}")
        return ds
        
    except Exception as e:
        logger.error(f"xarray Dataset creation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Test preprocessing pipeline
    try:
        # Sample paths (you'll need to adjust these)
        test_bands = {
            'B02': Path('data/sentinel/B02.tif'),
            'B03': Path('data/sentinel/B03.tif'),
            'B04': Path('data/sentinel/B04.tif'),
            'B08': Path('data/sentinel/B08.tif')
        }
        qa60_path = Path('data/sentinel/QA60.tif')
        
        # Test cloud masking
        logger.info("Testing cloud masking...")
        clear_mask = mask_clouds(qa60_path)
        
        # Test atmospheric correction
        logger.info("Testing atmospheric correction...")
        corrected_bands = correct_atmosphere(test_bands)
        
        # Test index calculation
        logger.info("Testing water index calculation...")
        indices = calculate_indices(corrected_bands)
        
        # Test water masking (combine with cloud mask)
        logger.info("Testing water masking...")
        water_mask = mask_water(indices) & clear_mask
        
        # Test feature extraction
        logger.info("Testing feature extraction...")
        features = extract_features(corrected_bands, indices, water_mask)
        
        # Create xarray dataset
        logger.info("Creating xarray dataset...")
        with rasterio.open(test_bands['B02']) as src:
            transform = src.transform
            crs = src.crs
        
        dataset = create_xarray_dataset(
            corrected_bands, indices, water_mask,
            transform, crs
        )
        
        # Save results
        output_dir = Path('data/processed')
        output_dir.mkdir(exist_ok=True)
        
        # Save as netCDF
        dataset.to_netcdf(output_dir / 'processed_data.nc')
        
        # Save feature array
        np.save(output_dir / 'features.npy', features)
        
        logger.info("Preprocessing pipeline test completed successfully!")
        logger.info(f"Final feature array shape: {features.shape}")
        logger.info(f"Outputs saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline test failed: {str(e)}")