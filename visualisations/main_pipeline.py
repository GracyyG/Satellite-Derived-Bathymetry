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

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.notebooks_dir = project_root / 'notebooks'
        self.config_path = project_root / 'config' / 'location_config.json'
        
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
        
        # Define notebook sequence - only include existing notebooks
        self.notebooks = [
            "01_select_location.ipynb",
            "01_sentinel2_download.ipynb",
            "02_data_preprocessing.ipynb",
            "03_band_extraction.ipynb",
            "03_model_training.ipynb",
            "04_model_validation.ipynb",
            "05_model_deployment.ipynb",
            "06_visualization.ipynb",
            "07_icesat_integration.ipynb",
            "08_model_optimization.ipynb",
            "09_final_visual_showcase.ipynb"
        ]
    
    def print_header(self):
        """Print pipeline header with region info"""
        print("\n" + "="*60)
        print(f"[3D] Running SDB pipeline for region: {self.region}")
        print("Area of Interest:")
        print(f"  Latitude:  {self.config['aoi']['min_lat']}° to {self.config['aoi']['max_lat']}°")
        print(f"  Longitude: {self.config['aoi']['min_lon']}° to {self.config['aoi']['max_lon']}°")
        print("="*60 + "\n")
    
    def execute_notebook(self, notebook_name: str) -> bool:
        """Execute a single notebook and return success status"""
        input_path = self.notebooks_dir / notebook_name
        output_path = self.executed_dir / notebook_name
        
        # Check if notebook exists before attempting execution
        if not input_path.exists():
            self.logger.warning(f"[WARN] {notebook_name} not found at {input_path} - skipping")
            print(f"[WARN] Skipping: {notebook_name} (not found)")
            return True  # Don't fail the pipeline for missing notebooks
        
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
        
        executed_count = 0
        failed_count = 0
        
        for notebook in self.notebooks:
            input_path = self.notebooks_dir / notebook
            if input_path.exists():
                if self.execute_notebook(notebook):
                    executed_count += 1
                else:
                    failed_count += 1
                    # Continue with other notebooks instead of stopping
                    self.logger.warning(f"Continuing pipeline despite failure in {notebook}")
            else:
                self.logger.info(f"Skipping {notebook} - not found")
        
        # Pipeline completed - report results
        duration = datetime.now() - pipeline_start
        
        if executed_count > 0:
            success_msg = f"""
[OK] Pipeline completed for region: {self.region}
   Total duration: {duration}
   Notebooks executed: {executed_count}
   Notebooks failed: {failed_count}
   Notebooks skipped: {len(self.notebooks) - executed_count - failed_count}
   
   Outputs available in:
   - Executed notebooks: {self.executed_dir}
   - Final results: {self.showcase_dir}
   - Logs: {self.log_dir}
"""
            print(success_msg)
            self.logger.info(f"Pipeline completed in {duration} - {executed_count} successful, {failed_count} failed")
            
            # Run visualization scripts after notebooks complete
            try:
                self.run_visualization_scripts()
            except Exception as e:
                self.logger.warning(f"Visualization scripts failed: {str(e)}")
                print("[WARN] Visualization scripts failed - check logs for details")
            return True
        else:
            self.logger.error("No notebooks were successfully executed")
            return False
    
    def run_visualization_scripts(self):
        """Run visualization scripts after pipeline completion"""
        viz_dir = self.project_root / "visualisations"
        viz_scripts = [
            "comprehensive_model_comparison.py",
            "depth_stratified_3d_analysis.py", 
            "geographic_heatmap_analysis.py",
            "mesh_surface_3d_analysis.py"
        ]
        
        self.logger.info("Running visualization scripts...")
        print("\n[3D] Running visualization scripts...")
        
        for script in viz_scripts:
            script_path = viz_dir / script
            if script_path.exists():
                try:
                    import subprocess
                    result = subprocess.run([
                        sys.executable, str(script_path)
                    ], cwd=str(self.project_root), 
                       capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        self.logger.info(f"[OK] {script} completed successfully")
                        print(f"[OK] {script} completed")
                    else:
                        self.logger.warning(f"[ERROR] {script} failed: {result.stderr}")
                        print(f"[ERROR] {script} failed")
                except Exception as e:
                    self.logger.warning(f"[ERROR] {script} failed: {str(e)}")
                    print(f"[ERROR] {script} failed: {str(e)}")
            else:
                self.logger.info(f"Skipping {script} - not found")

def main():
    """Main entry point"""
    try:
        # Get project root - go up one level from visualisations/
        project_root = Path(__file__).parent.parent
        
        # Initialize and run pipeline
        pipeline = SDBPipeline(project_root)
        success = pipeline.run_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"[ERROR] Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()