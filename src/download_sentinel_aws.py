"""
Sentinel-2 Data Download Module using AWS Open Data

This module handles downloading Sentinel-2 imagery from AWS Open Data registry.
No authentication required as this is a public dataset.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import logging
import boto3
import rasterio
from rasterio.vrt import WarpedVRT
import requests
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sentinel_aws_path(tile_id: str, date: datetime) -> str:
    """
    Convert tile ID and date to AWS S3 path.
    
    Args:
        tile_id: Sentinel-2 tile ID (e.g., '10SEG')
        date: Acquisition date
        
    Returns:
        S3 path to the Sentinel-2 data
    """
    utm_zone = tile_id[:2]
    lat_band = tile_id[2]
    square = tile_id[3:]
    
    return f"tiles/{utm_zone}/{lat_band}/{square}/{date.strftime('%Y/%m/%d')}"

def find_sentinel_scenes(
    tile_id: str,
    start_date: datetime,
    end_date: datetime,
    max_cloud_cover: float = 20.0
) -> List[Dict]:
    """
    Find Sentinel-2 scenes on AWS for given parameters.
    
    Args:
        tile_id: Sentinel-2 tile ID
        start_date: Start date for search
        end_date: End date for search
        max_cloud_cover: Maximum cloud cover percentage
        
    Returns:
        List of dictionaries containing scene metadata
    """
    try:
        # Configure S3 client for public access
        s3 = boto3.client(
            's3',
            region_name='eu-central-1',
            config=boto3.Config(signature_version=boto3.UNSIGNED)
        )
        scenes = []
        
        # List objects in the S3 bucket
        bucket = 'sentinel-s2-l2a'
        prefix = get_sentinel_aws_path(tile_id, start_date)
        
        logger.info(f"Searching for scenes in tile {tile_id}")
        
        paginator = s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                # Check if this is a metadata file
                if obj['Key'].endswith('metadata.json'):
                    # Get metadata
                    response = s3.get_object(Bucket=bucket, Key=obj['Key'])
                    metadata = json.loads(response['Body'].read())
                    
                    # Check cloud cover
                    cloud_cover = float(metadata.get('cloudyPixelPercentage', 100))
                    if cloud_cover > max_cloud_cover:
                        continue
                    
                    scenes.append({
                        'tile_id': tile_id,
                        'date': metadata.get('datetime'),
                        'cloud_cover': cloud_cover,
                        'aws_path': f"s3://{bucket}/{obj['Key'].rsplit('/', 1)[0]}",
                        'metadata': metadata
                    })
        
        logger.info(f"Found {len(scenes)} scenes matching criteria")
        return scenes
        
    except Exception as e:
        logger.error(f"Scene search failed: {str(e)}")
        raise

def download_bands(
    scene_path: str,
    output_dir: Path,
    bands: List[str] = ['B02', 'B03', 'B04', 'B08'],
    resolution: int = 10
) -> Dict[str, Path]:
    """
    Download specific bands from a Sentinel-2 scene on AWS.
    
    Args:
        scene_path: AWS S3 path to the scene
        output_dir: Directory to save downloaded bands
        bands: List of band names to download
        resolution: Spatial resolution in meters (10, 20, or 60)
        
    Returns:
        Dictionary mapping band names to their file paths
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        band_paths = {}
        
        # Remove s3:// prefix
        scene_path = scene_path.replace('s3://', '')
        bucket = scene_path.split('/')[0]
        prefix = '/'.join(scene_path.split('/')[1:])
        
        # Configure S3 client for public access
        s3 = boto3.client(
            's3',
            region_name='eu-central-1',
            config=boto3.Config(signature_version=boto3.UNSIGNED)
        )
        
        for band in bands:
            # Construct band path
            res_folder = f"R{resolution}m"
            band_filename = f"{band}_{resolution}m.jp2"
            band_key = f"{prefix}/{res_folder}/{band_filename}"
            
            # Download band
            output_path = output_dir / band_filename
            logger.info(f"Downloading {band} to {output_path}")
            
            s3.download_file(bucket, band_key, str(output_path))
            band_paths[band] = output_path
            
        return band_paths
        
    except Exception as e:
        logger.error(f"Band download failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    tile_id = "10SEG"  # Example tile ID
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    try:
        # Search for scenes
        scenes = find_sentinel_scenes(tile_id, start_date, end_date)
        
        if scenes:
            # Download bands from first scene
            scene = scenes[0]
            output_dir = Path("data/sentinel")
            band_paths = download_bands(scene['aws_path'], output_dir)
            
            logger.info("Download completed successfully!")
            for band, path in band_paths.items():
                logger.info(f"{band}: {path}")
                
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")