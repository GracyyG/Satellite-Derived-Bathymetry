#!/usr/bin/env python3
"""
Satellite-Derived Bathymetry (SDB) Pipeline Runner

This script executes the complete SDB workflow by running all notebooks
in sequence. It handles logging, error handling, and output organization.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import papermill as pm
from typing import List, Dict, Optional

# Configure logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'pipeline_log.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class SDBPipeline:
    """Manages the execution of the SDB notebook pipeline"""
    
    def check_api_credentials(self) -> Optional[dict]:
        """
        Check for valid Copernicus API credentials
        Returns credentials dict if valid, None if not found or invalid
        """
        config_paths = [
            self.project_root.parent / 'sentinel2_pipeline' / 'config' / 'sentinel_api_config.json',
            self.project_root / 'config' / 'sentinel_api_config.json'
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        creds = json.load(f)
                    
                    if not creds.get('client_id') or not creds.get('client_secret'):
                        self.logger.error(f"[ERROR] Missing required credentials in {config_path}")
                        continue
                    
                    self.logger.info("[OK] Copernicus API credentials verified and loaded from sentinel_api_config.json")
                    return creds
                    
                except Exception as e:
                    self.logger.error(f"[ERROR] Error loading credentials from {config_path}: {str(e)}")
                    continue
        
        self.logger.error("[ERROR] No valid Copernicus API credentials found")
        return None

    def __init__(self, project_root: Path, run_mode: str = "fast"):
        self.project_root = project_root
        self.notebooks_dir = project_root / 'notebooks'
        self.config_path = project_root / 'config' / 'location_config.json'
        self.run_mode = run_mode
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        # Set up paths
        self.region = self.config['region_name']
        self.region_slug = self.region.lower().replace(' ', '_')
        self.output_dir = project_root / 'outputs' / self.region_slug
        self.executed_dir = self.output_dir / 'executed_notebooks'
        self.showcase_dir = self.output_dir / 'final_showcase'
        self.log_dir = project_root / 'logs'
        
        # Create directories
        for dir_path in [self.output_dir, self.executed_dir, 
                        self.showcase_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(self.log_dir)
        
        # Define notebook sequence based on run mode
        if run_mode == "fast":
            # Fast mode: ONLY visualization - no preprocessing! (30 seconds max)
            self.notebooks = [
                "06_visualization.ipynb"
            ]
        else:
            # Full mode: Complete ML pipeline (for training/research)
            self.notebooks = [
                # "01_sentinel2_download.ipynb",  # Skipping due to Unicode encoding issues
                "01_select_location.ipynb",
                "02_data_preprocessing.ipynb",
                "03_band_extraction.ipynb",
                "03_model_training.ipynb",
                "04_model_validation.ipynb",
                "05_model_deployment.ipynb",
                "06_visualization.ipynb",
                "07_icesat_integration.ipynb",
                "08_model_optimization.ipynb"
            ]
    
    def print_header(self):
        """Print pipeline header with region info"""
        print("\n" + "="*60)
        print(f"[{self.run_mode.upper()}] Running SDB pipeline for region: {self.region}")
        print("Area of Interest:")
        print(f"  Latitude:  {self.config['aoi']['min_lat']}° to {self.config['aoi']['max_lat']}°")
        print(f"  Longitude: {self.config['aoi']['min_lon']}° to {self.config['aoi']['max_lon']}°")
        if self.run_mode == "fast":
            print("Mode: FAST - Visualization only (models must exist)")
        else:
            print("Mode: FULL - Complete ML pipeline with training")
        print("="*60 + "\n")
    
    def execute_notebook(self, notebook_name: str) -> bool:
        """Execute a single notebook and return success status"""
        input_path = self.notebooks_dir / notebook_name
        output_path = self.executed_dir / notebook_name
        
        start_time = datetime.now()
        self.logger.info(f"Starting notebook: {notebook_name}")
        print(f"[RUN] Executing: {notebook_name}")
        
        try:
            # Execute notebook
            pm.execute_notebook(
                str(input_path),
                str(output_path),
                kernel_name='python3',  # Specify kernel explicitly
                parameters={
                    'region_name': self.region,
                    'output_dir': str(self.output_dir)
                }
            )
            
            # Log success
            duration = datetime.now() - start_time
            self.logger.info(f"[OK] {notebook_name} completed successfully in {duration}")
            print(f"[OK] Completed: {notebook_name} ({duration})")
            return True
            
        except Exception as e:
            # Log failure
            duration = datetime.now() - start_time
            self.logger.error(f"[ERROR] {notebook_name} failed after {duration}: {str(e)}")
            print(f"[ERROR] {notebook_name} failed - see logs for details")
            return False
    
    def run_pipeline(self):
        """Execute complete notebook pipeline"""
        self.print_header()
        pipeline_start = datetime.now()
        
        # Check Copernicus API credentials before starting
        if not self.check_api_credentials():
            self.logger.warning("Skipping Sentinel-2 download due to missing credentials")
            print("[WARN] No valid Copernicus API credentials found - skipping Sentinel-2 download")
        
        for notebook in self.notebooks:
            if not self.execute_notebook(notebook):
                self.logger.error("Pipeline execution stopped due to error")
                return False
        
        # Pipeline completed successfully
        duration = datetime.now() - pipeline_start
        success_msg = f"""
[OK] Pipeline completed successfully for region: {self.region}
   Total duration: {duration}
   
   Outputs available in:
   - Executed notebooks: {self.executed_dir}
   - Final results: {self.showcase_dir}
   - Logs: {self.log_dir}
"""
        print(success_msg)
        self.logger.info(f"Pipeline completed in {duration}")
        return True

def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Satellite-Derived Bathymetry Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Run Modes:
  fast  - Quick visualization generation (requires existing models)
          Runs: location setup → preprocessing → visualization (2-5 minutes)
          
  full  - Complete ML pipeline with model training
          Runs: all notebooks including training/validation (45-120 minutes)
          
Examples:
  python main_pipeline.py --mode fast    # Quick demo/visualization
  python main_pipeline.py --mode full    # Full training pipeline
            """
        )
        parser.add_argument(
            "--mode", 
            choices=["fast", "full"], 
            default="fast",
            help="Pipeline run mode (default: fast)"
        )
        args = parser.parse_args()
        
        # Get project root
        project_root = Path(__file__).parent
        
        # Initialize and run pipeline
        pipeline = SDBPipeline(project_root, run_mode=args.mode)
        success = pipeline.run_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"[ERROR] Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()