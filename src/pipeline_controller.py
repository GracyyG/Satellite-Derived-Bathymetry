#!/usr/bin/env python3
"""
Satellite-Derived Bathymetry (SDB) Pipeline Controller

This script orchestrates the complete SDB pipeline by:
1. Loading the location configuration
2. Setting up region-specific output directories
3. Running each pipeline stage in sequence
4. Logging progress and results
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import pipeline modules
from sdb_project.src.download_sentinel_aws import find_sentinel_scenes, download_bands
from datetime import datetime
import yaml
from sdb_project.src.main import SDBPipeline
import logging

def setup_logging(region_name):
    """Configure logging with region-specific output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pipeline_{region_name.lower().replace(' ', '_')}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_override=None):
    """
    Load location and pipeline configurations.
    
    Args:
        config_override (dict): Optional dictionary with AOI coordinates to override config file
    """
    # Load location config
    location_config_path = project_root / 'config' / 'location_config.json'
    
    if config_override:
        location_config = {
            "region_name": config_override.get("region_name", "Custom_Region"),
            "aoi": {
                "min_lat": config_override["min_lat"],
                "max_lat": config_override["max_lat"],
                "min_lon": config_override["min_lon"],
                "max_lon": config_override["max_lon"]
            }
        }
    else:
        if not location_config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {location_config_path}. "
                "Please run 01_select_location.ipynb first."
            )
        
        with open(location_config_path) as f:
            location_config = json.load(f)
    
    # Load pipeline config 
    yaml_config_path = project_root / 'config.yaml'
    with open(yaml_config_path) as f:
        config = yaml.safe_load(f)
    
    # Update config with location info
    config['area'] = {
        'name': location_config['region_name'],
        'bbox': [
            location_config['aoi']['min_lon'],
            location_config['aoi']['min_lat'], 
            location_config['aoi']['max_lon'],
            location_config['aoi']['max_lat']
        ]
    }
    
    return config

def run_sdb_pipeline(config, logger):
    """Run the complete SDB pipeline using the SDBPipeline class."""
    logger.info("Starting SDB pipeline...")
    
    try:
        pipeline = SDBPipeline(config)

        # Run the complete pipeline
        area_name = config['area']['name']
        bbox = config['area']['bbox']
        start_date = config['start_date'] 
        end_date = config['end_date']
        
        logger.info("Processing data and training models...")
        results = pipeline.run(
            area_name=area_name,
            bbox=bbox, 
            start_date=start_date,
            end_date=end_date
        )
        
        if results:
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {results['predictions_path']}")
            if results['metrics']:
                logger.info("Performance metrics:")
                for metric, value in results['metrics'].items():
                    logger.info(f"{metric}: {value:.3f}")
        else:
            logger.error("Pipeline failed. Check logs for details.")
            
        return results is not None
            
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        return False

def main(args=None):
    """Main pipeline execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SDB Pipeline Controller")
    parser.add_argument("--min-lat", type=float, help="Minimum latitude")
    parser.add_argument("--max-lat", type=float, help="Maximum latitude")
    parser.add_argument("--min-lon", type=float, help="Minimum longitude")
    parser.add_argument("--max-lon", type=float, help="Maximum longitude")
    parser.add_argument("--region-name", type=str, help="Region name for custom AOI")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args(args)
    
    # Check if AOI override is provided
    config_override = None
    if all([args.min_lat, args.max_lat, args.min_lon, args.max_lon]):
        config_override = {
            "region_name": args.region_name or "Custom_Region",
            "min_lat": args.min_lat,
            "max_lat": args.max_lat,
            "min_lon": args.min_lon,
            "max_lon": args.max_lon
        }
    
    # Load configuration
    config = load_config(config_override)
    area_name = config['area']['name']
    
    # Add dates to config
    config['start_date'] = args.start_date or datetime.now().strftime("%Y-%m-%d")
    config['end_date'] = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    # Setup logging
    logger = setup_logging(area_name)
    logger.info(f"Starting SDB pipeline for region: {area_name}")
    
    # Run pipeline
    success = run_sdb_pipeline(config, logger)
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)