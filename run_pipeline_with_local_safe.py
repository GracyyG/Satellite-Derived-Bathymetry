#!/usr/bin/env python3
"""
Satellite-Derived Bathymetry (SDB) Pipeline Runner - Local SAFE Fallback

This script executes the complete SDB workflow using existing local SAFE files,
skipping network downloads. It detects available SAFE files and runs all 
downstream notebooks in sequence.
"""

import os
import sys
import json
import logging
import glob
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

class SDBPipelineLocalSafe:
    """Manages the execution of the SDB notebook pipeline using local SAFE files"""
    
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
        
        # Define notebook sequence (EXCLUDING 01_sentinel2_download.ipynb)
        self.notebooks = [
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
        
        # Initialize SAFE file detection
        self.use_local_safe = False
        self.selected_safe_path = None
    
    def find_safe_files(self) -> List[Path]:
        """
        Search for existing SAFE files in multiple potential locations
        Returns list of Path objects for found SAFE files
        """
        search_paths = [
            # Primary path: data/sentinel/{region_name}/raw/
            self.project_root / 'data' / 'sentinel' / self.region_slug / 'raw',
            # Alternative absolute paths
            Path('D:/Project/sentinel2_pipeline/data/sentinel'),
            Path('D:/Project/sentinel2_pipeline/data/sentinel') / self.region_slug,
            Path('D:/Project/sentinel2_pipeline/data/sentinel') / self.region_slug / 'raw',
            # Parent project sentinel2_pipeline paths
            self.project_root.parent / 'sentinel2_pipeline' / 'data' / 'sentinel',
            # Any other data/sentinel folders in project tree
            self.project_root / 'data' / 'sentinel',
        ]
        
        safe_files = []
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            self.logger.info(f"Searching for SAFE files in: {search_path}")
            
            # Look for .SAFE directories
            safe_dirs = list(search_path.glob('**/*.SAFE'))
            
            # Look for .zip files containing SAFE (case insensitive)
            zip_files = []
            for pattern in ['**/*SAFE*.zip', '**/*safe*.zip', '**/*S2*.zip', '**/*s2*.zip']:
                zip_files.extend(search_path.glob(pattern))
            
            # Combine and add to results
            found_files = safe_dirs + zip_files
            if found_files:
                self.logger.info(f"Found {len(found_files)} SAFE file(s) in {search_path}")
                for f in found_files:
                    self.logger.info(f"  - {f.name}")
                safe_files.extend(found_files)
        
        return safe_files
    
    def select_best_safe_file(self, safe_files: List[Path]) -> Optional[Path]:
        """
        Select the most recent/best SAFE file from the list
        Returns Path to selected file or None
        """
        if not safe_files:
            return None
        
        # Sort by modification time (most recent first)
        try:
            safe_files_with_time = [(f, f.stat().st_mtime) for f in safe_files]
            safe_files_with_time.sort(key=lambda x: x[1], reverse=True)
            selected = safe_files_with_time[0][0]
            
            self.logger.info(f"Selected most recent SAFE file: {selected}")
            return selected
            
        except Exception as e:
            self.logger.warning(f"Error selecting SAFE file: {e}, using first available")
            return safe_files[0]
    
    def setup_safe_fallback(self):
        """
        Detect and configure local SAFE file usage
        Sets use_local_safe flag and selected_safe_path
        """
        self.logger.info("🔍 Searching for existing SAFE files...")
        print("🔍 Searching for existing SAFE files...")
        
        # Find available SAFE files
        safe_files = self.find_safe_files()
        
        if not safe_files:
            self.logger.error("❌ No Sentinel SAFE files found locally")
            print("❌ No Sentinel SAFE files found locally. Please download or provide path to SAFE file.")
            return False
        
        # Select best SAFE file
        self.selected_safe_path = self.select_best_safe_file(safe_files)
        
        if self.selected_safe_path:
            self.use_local_safe = True
            self.logger.info(f"✅ Using local SAFE file: {self.selected_safe_path}")
            print(f"✅ Using local SAFE file: {self.selected_safe_path}")
            return True
        
        return False
    
    def print_header(self):
        """Print pipeline header with region info"""
        print("\n" + "="*70)
        print(f"🌍 Running SDB Pipeline (Local SAFE Mode) for: {self.region}")
        print("Area of Interest:")
        print(f"  Latitude:  {self.config['aoi']['min_lat']}° to {self.config['aoi']['max_lat']}°")
        print(f"  Longitude: {self.config['aoi']['min_lon']}° to {self.config['aoi']['max_lon']}°")
        if self.use_local_safe:
            print(f"SAFE File: {self.selected_safe_path}")
        print("="*70 + "\n")
    
    def execute_notebook(self, notebook_name: str) -> bool:
        """Execute a single notebook and return success status"""
        input_path = self.notebooks_dir / notebook_name
        output_path = self.executed_dir / notebook_name
        
        if not input_path.exists():
            self.logger.warning(f"⚠️  Notebook not found: {notebook_name}, skipping...")
            print(f"⚠️  Skipping missing notebook: {notebook_name}")
            return True  # Don't fail pipeline for missing optional notebooks
        
        start_time = datetime.now()
        self.logger.info(f"Starting notebook: {notebook_name}")
        print(f"🔄 Executing: {notebook_name}")
        
        try:
            # Prepare parameters
            parameters = {
                'region_name': self.region,
                'output_dir': str(self.output_dir)
            }
            
            # Add SAFE file path for preprocessing notebooks
            if self.use_local_safe and notebook_name.startswith('02_'):
                parameters['safe_file_path'] = str(self.selected_safe_path)
            
            # Execute notebook
            pm.execute_notebook(
                str(input_path),
                str(output_path),
                kernel_name='python3',
                parameters=parameters,
                progress_bar=False,  # Reduce output noise
                request_save_on_cell_execute=True
            )
            
            # Log success
            duration = datetime.now() - start_time
            self.logger.info(f"✅ {notebook_name} completed successfully in {duration}")
            print(f"✅ Completed: {notebook_name} ({duration})")
            return True
            
        except Exception as e:
            # Log failure
            duration = datetime.now() - start_time
            self.logger.error(f"❌ {notebook_name} failed after {duration}: {str(e)}")
            print(f"❌ {notebook_name} failed - see logs for details")
            print(f"   Error: {str(e)[:100]}...")
            return False
    
    def copy_safe_to_expected_location(self):
        """
        Copy/link SAFE file to expected data directory structure
        This ensures downstream notebooks can find the data
        """
        if not self.use_local_safe:
            return
        
        # Expected location for the region
        expected_dir = self.project_root / 'data' / 'sentinel' / self.region_slug / 'raw'
        expected_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if SAFE is already in expected location
        if self.selected_safe_path.parent == expected_dir:
            self.logger.info("SAFE file already in expected location")
            return
        
        # Create a symlink or copy reference
        try:
            expected_path = expected_dir / self.selected_safe_path.name
            
            if not expected_path.exists():
                # Create symbolic link on Windows (requires admin) or copy on other systems
                if sys.platform.startswith('win'):
                    try:
                        expected_path.symlink_to(self.selected_safe_path, target_is_directory=self.selected_safe_path.is_dir())
                        self.logger.info(f"Created symlink: {expected_path} -> {self.selected_safe_path}")
                    except (OSError, NotImplementedError):
                        # Fallback: create a reference file
                        with open(expected_path.with_suffix('.safe_reference'), 'w') as f:
                            f.write(str(self.selected_safe_path))
                        self.logger.info(f"Created reference file: {expected_path.with_suffix('.safe_reference')}")
                else:
                    expected_path.symlink_to(self.selected_safe_path)
                    self.logger.info(f"Created symlink: {expected_path} -> {self.selected_safe_path}")
        
        except Exception as e:
            self.logger.warning(f"Could not create link to SAFE file: {e}")
    
    def run_pipeline(self):
        """Execute complete notebook pipeline with local SAFE fallback"""
        
        # Step 1: Setup SAFE fallback
        if not self.setup_safe_fallback():
            return False
        
        # Step 2: Copy SAFE to expected location if needed
        self.copy_safe_to_expected_location()
        
        # Step 3: Print header and start pipeline
        self.print_header()
        pipeline_start = datetime.now()
        
        self.logger.info(f"Starting pipeline execution with {len(self.notebooks)} notebooks")
        
        # Step 4: Execute notebooks in sequence
        failed_notebooks = []
        for i, notebook in enumerate(self.notebooks, 1):
            print(f"\n[{i}/{len(self.notebooks)}] {notebook}")
            if not self.execute_notebook(notebook):
                failed_notebooks.append(notebook)
                # Continue with remaining notebooks rather than stopping
        
        # Step 5: Report results
        duration = datetime.now() - pipeline_start
        
        if failed_notebooks:
            self.logger.warning(f"Pipeline completed with {len(failed_notebooks)} failed notebook(s): {failed_notebooks}")
            print(f"\n⚠️  Pipeline completed with errors. Failed notebooks: {failed_notebooks}")
        else:
            self.logger.info("Pipeline completed successfully - all notebooks executed")
            print(f"\n✅ Pipeline completed successfully!")
        
        # Final success message
        success_msg = f"""
📊 Pipeline Summary for {self.region}:
   Duration: {duration}
   SAFE File: {self.selected_safe_path}
   
   📁 Output Locations:
   - Executed notebooks: {self.executed_dir}
   - Final showcase: {self.showcase_dir}
   - Logs: {self.log_dir}
   
✅ Visuals created: {self.showcase_dir}/
"""
        print(success_msg)
        self.logger.info(f"Pipeline summary - Duration: {duration}, Outputs: {self.showcase_dir}")
        
        return len(failed_notebooks) == 0

def main():
    """Main entry point"""
    try:
        # Get project root
        project_root = Path(__file__).parent
        
        print("🚀 Starting SDB Pipeline with Local SAFE Fallback")
        print(f"📂 Project root: {project_root}")
        
        # Initialize and run pipeline
        pipeline = SDBPipelineLocalSafe(project_root)
        success = pipeline.run_pipeline()
        
        if success:
            print("\n🎉 Pipeline execution completed successfully!")
        else:
            print("\n⚠️  Pipeline execution completed with some issues - check logs")
        
        # Exit with appropriate code  
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()