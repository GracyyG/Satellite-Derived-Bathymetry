"""
Satellite-Derived Bathymetry (SDB) Pipeline

This script orchestrates the complete SDB workflow:
1. Download Sentinel-2 imagery
2. Preprocess the data and extract features
3. Train and evaluate ML models
4. Visualize and export results

Usage:
    python main.py --config path/to/config.yaml
    python main.py --area "area_name" --start-date "YYYY-MM-DD" --end-date "YYYY-MM-DD"
"""

import argparse
import logging
from pathlib import Path
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import rasterio
from rasterio.windows import Window
import math
from mgrs import MGRS

from sdb_project.src.download_sentinel_aws import find_sentinel_scenes, download_bands
# No preprocessing needed - using SentinelHub's preproocessed data
from sdb_project.src.sdb_model import SDBModel 
from sdb_project.src import visualize
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SDBPipeline:
    """Main class for running the complete SDB workflow."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SDB pipeline.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = SDBModel()
        
    def get_mgrs_tile(self, lat: float, lon: float) -> str:
        """
        Convert latitude/longitude coordinates to MGRS tile ID.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            
        Returns:
            MGRS tile ID (e.g., '43QBA')
        """
        mgrs = MGRS()
        tile = mgrs.toMGRS(lat, lon, MGRSPrecision=0)
        return tile[:5]  # First 5 characters give the 100km grid square

    def run(
        self,
        area_name: str,
        bbox: tuple,
        start_date: str,
        end_date: str,
        reference_data: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run the complete SDB pipeline.
        
        Args:
            area_name: Name of the study area
            bbox: Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date for image search (YYYY-MM-DD)
            end_date: End date for image search (YYYY-MM-DD)
            reference_data: Optional path to reference bathymetry data
            
        Returns:
            Dictionary containing results and performance metrics
        """
        try:
            logger.info(f"Starting SDB pipeline for {area_name}")
            
            # 1. Download Sentinel-2 data
            # First find available scenes
            # Get central coordinates and find MGRS tile
            center_lat = (bbox[1] + bbox[3]) / 2  # Average of min/max lat
            center_lon = (bbox[0] + bbox[2]) / 2  # Average of min/max lon
            tile_id = self.get_mgrs_tile(center_lat, center_lon)
            
            scenes = find_sentinel_scenes(
                tile_id=tile_id,
                start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(end_date, "%Y-%m-%d")
            )
            
            if not scenes:
                raise RuntimeError("No suitable Sentinel-2 scenes found")
                
            # Download the first good scene
            scene = scenes[0]
            image_bands = download_bands(
                scene_path=scene['aws_path'],
                output_dir=self.output_dir / 'sentinel_data'
            )
            
            # 2. Extract features and preprocess
            # Process first band to get coordinates
            first_band_path = list(image_bands.values())[0]
            with rasterio.open(first_band_path) as src:
                # Get dimensions
                height = src.height
                width = src.width
                
                # Get transformation matrix
                transform = src.transform
                
                # Create coordinate arrays
                # Create arrays of coordinates
                coordinates = np.zeros((height, width, 2))
                for i in range(height):
                    for j in range(width):
                        coordinates[i,j] = transform * (j, i)  # Convert pixel to coordinate
                
                # Extract bands
                features = []
                for band_path in image_bands.values():
                    with rasterio.open(band_path) as band_src:
                        band_data = band_src.read(1)
                        features.append(band_data)
                
                features = np.stack(features)
                        
            # 3. Apply ML models
            if reference_data:
                # Training mode
                reference = np.load(reference_data)
                predictions = self.model.predict(features)
                metrics = {
                    'rmse': np.sqrt(np.mean((predictions - reference) ** 2)),
                    'r2': np.corrcoef(predictions.ravel(), reference.ravel())[0, 1] ** 2
                }
            else:
                # Prediction mode 
                predictions = self.model.predict(features)
                metrics = {}
            
            # 4. Visualize results
            viz_dir = self.output_dir / 'visualizations'
            
            # 2D bathymetry plot
            visualize.plot_bathymetry_2d(
                depths=predictions,
                coordinates=coordinates,
                output_path=viz_dir / f'{area_name}_bathymetry_2d.png'
            )
            
            # Interactive 3D plot
            visualize.plot_bathymetry_3d(
                depths=predictions,
                coordinates=coordinates,
                output_path=viz_dir / f'{area_name}_bathymetry_3d.png'
            )
            
            # Error analysis if reference data available
            if reference_data:
                visualize.plot_error_distribution(
                    predicted=predictions,
                    reference=reference,
                    output_path=viz_dir / f'{area_name}_error_distribution.png'
                )
                
                visualize.plot_comparison(
                    predicted=predictions,
                    reference=reference,
                    output_path=viz_dir / f'{area_name}_comparison.png'
                )
            
            # Save results
            np.save(self.output_dir / 'results' / f'{area_name}_predictions.npy', predictions)
            
            logger.info(f"SDB pipeline completed for {area_name}")
            return {
                'area_name': area_name,
                'predictions_path': str(self.output_dir / 'results' / f'{area_name}_predictions.npy'),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main entry point for the SDB pipeline."""
    parser = argparse.ArgumentParser(description="Run Satellite-Derived Bathymetry pipeline")
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--area', type=str, help='Name of the study area')
    parser.add_argument('--bbox', type=str, help='Bounding box coordinates (min_lon,min_lat,max_lon,max_lat)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--reference', type=str, help='Path to reference bathymetry data')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default configuration
        config = {
            'output_dir': 'output',
            'copernicus': {
                'username': 'your_username',
                'password': 'your_password'
            },
            'preprocessing': {
                'atmospheric_correction': True,
                'water_mask_threshold': 0.3
            },
            'model': {
                'type': 'random_forest',
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': 20
                }
            }
        }
    
    # Parse bbox if provided
    if args.bbox:
        bbox = tuple(map(float, args.bbox.split(',')))
    else:
        bbox = config.get('area', {}).get('bbox')
    
    # Initialize and run pipeline
    pipeline = SDBPipeline(config)
    results = pipeline.run(
        area_name=args.area,
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        reference_data=args.reference if args.reference else None
    )
    
    # Print results summary
    print("\nPipeline Results:")
    print(f"Area: {results['area_name']}")
    print(f"Predictions saved to: {results['predictions_path']}")
    if results['metrics']:
        print("\nPerformance Metrics:")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.3f}")

if __name__ == '__main__':
    main()