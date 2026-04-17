#!/usr/bin/env python3
"""
Auto-Process All SAFE Regions Script
===================================
Automatically processes all regions with SAFE files through the complete SDB pipeline.

This script:
- Discovers all regions with SAFE files
- Runs the full pipeline for each region
- Avoids reprocessing already completed regions
- Handles errors gracefully per region
- Provides comprehensive logging and progress tracking

Usage:
    python process_all_regions.py [--force] [--region <name>]

Options:
    --force         Force reprocessing even if region already processed
    --region <name> Process only the specified region
    --dry-run       Show what would be processed without actually running

Author: SDB Pipeline Team
Date: November 13, 2025
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Import pipeline components
import papermill as pm
from src.safe_file_manager import SAFEFileManager
from auto_safe_config import AutoSAFEConfigurator

class RegionProcessor:
    """Handles processing of multiple regions with SAFE files"""
    
    def __init__(self, force_reprocess=False, dry_run=False):
        self.force_reprocess = force_reprocess
        self.dry_run = dry_run
        self.results = {
            'processed': [],
            'skipped': [],
            'failed': []
        }
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.safe_manager = SAFEFileManager()
        self.configurator = AutoSAFEConfigurator()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"process_all_regions_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== AUTO-PROCESS ALL REGIONS STARTED ===")
    
    def discover_regions(self) -> List[Tuple[str, Path]]:
        """Discover all regions with SAFE files"""
        self.logger.info("Scanning for regions with SAFE files...")
        
        data_root = Path("data/sentinel")
        regions_with_safe = []
        
        if not data_root.exists():
            self.logger.error(f"Data root directory not found: {data_root}")
            return regions_with_safe
        
        for region_dir in data_root.iterdir():
            if not region_dir.is_dir():
                continue
                
            region_name = region_dir.name
            raw_dir = region_dir / "raw"
            
            # Check if raw directory exists
            if not raw_dir.exists():
                self.logger.debug(f"No raw/ directory found in {region_name}")
                continue
            
            # Check for SAFE files
            safe_files = []
            for item in raw_dir.iterdir():
                if item.is_file() and item.name.endswith('.zip'):
                    if self._is_safe_zip(item):
                        safe_files.append(item)
                elif item.is_dir() and item.name.endswith('.SAFE'):
                    safe_files.append(item)
            
            if safe_files:
                regions_with_safe.append((region_name, region_dir))
                self.logger.info(f"Found region '{region_name}' with {len(safe_files)} SAFE files")
            else:
                self.logger.debug(f"No SAFE files found in {region_name}/raw/")
        
        self.logger.info(f"Total regions with SAFE files: {len(regions_with_safe)}")
        return regions_with_safe
    
    def _is_safe_zip(self, zip_path: Path) -> bool:
        """Check if a zip file contains Sentinel-2 SAFE data"""
        return 'S2' in zip_path.name and 'MSIL' in zip_path.name
    
    def is_region_processed(self, region_name: str) -> bool:
        """Check if region has already been processed"""
        region_processed_dir = Path(f"data/sentinel/{region_name}/processed")
        
        if not region_processed_dir.exists():
            return False
        
        # Check for key processed directories
        required_dirs = ['bands', 'arrays']
        for req_dir in required_dirs:
            if not (region_processed_dir / req_dir).exists():
                return False
        
        # Check for output directory
        output_dir = Path(f"outputs/{region_name}")
        if not output_dir.exists():
            return False
        
        # Check for final showcase
        showcase_dir = output_dir / "final_showcase"
        if not showcase_dir.exists() or not any(showcase_dir.iterdir()):
            return False
        
        return True
    
    def configure_region(self, region_name: str, region_dir: Path) -> bool:
        """Configure SAFE files for a specific region"""
        try:
            # Create temporary config for this region
            temp_config = {
                "region_name": region_name,
                "aoi": self._get_default_aoi_for_region(region_name)
            }
            
            # Save temporary config
            config_path = Path("config/location_config.json")
            original_config = None
            if config_path.exists():
                with open(config_path, 'r') as f:
                    original_config = json.load(f)
            
            with open(config_path, 'w') as f:
                json.dump(temp_config, f, indent=2)
            
            # Run auto-configuration
            success = self.configurator.run_single_configuration()
            
            # Restore original config if it existed
            if original_config:
                with open(config_path, 'w') as f:
                    json.dump(original_config, f, indent=2)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to configure region {region_name}: {e}")
            return False
    
    def _get_default_aoi_for_region(self, region_name: str) -> Dict:
        """Get default AOI coordinates for a region"""
        # Default coordinate ranges for known regions
        region_coords = {
            'goa': {'min_lat': 15.0, 'max_lat': 15.8, 'min_lon': 73.7, 'max_lon': 74.3},
            'kachchh': {'min_lat': 22.5, 'max_lat': 24.0, 'min_lon': 68.0, 'max_lon': 71.0},
            'palk_strait': {'min_lat': 8.5, 'max_lat': 10.0, 'min_lon': 78.5, 'max_lon': 80.5},
            'rameswaram_palkstrait': {'min_lat': 8.5, 'max_lat': 10.0, 'min_lon': 78.5, 'max_lon': 80.5},
            'lakshadweep': {'min_lat': 10.5, 'max_lat': 12.5, 'min_lon': 71.5, 'max_lon': 74.0}
        }
        
        return region_coords.get(region_name.lower(), {
            'min_lat': 8.0, 'max_lat': 25.0, 'min_lon': 68.0, 'max_lon': 80.0
        })
    
    def run_pipeline_for_region(self, region_name: str) -> bool:
        """Execute the full SDB pipeline for a specific region"""
        self.logger.info(f"===== REGION: {region_name} =====")
        
        try:
            # Get SAFE file information
            safe_files_map = self.safe_manager.scan_safe_files()
            if region_name not in safe_files_map:
                self.logger.error(f"No SAFE files found for region: {region_name}")
                return False
            
            safe_file = safe_files_map[region_name][0]  # Use first/only SAFE file
            self.logger.info(f"SAFE file found: {safe_file}")
            
            processing_start = datetime.now()
            self.logger.info(f"Processing started at: {processing_start}")
            
            # Define notebook execution sequence
            notebooks = [
                "notebooks/02_data_preprocessing.ipynb",
                "notebooks/03_band_extraction.ipynb", 
                "notebooks/03_model_training.ipynb",
                "notebooks/04_model_validation.ipynb",
                "notebooks/06_visualization.ipynb",
                "notebooks/09_final_visual_showcase.ipynb"
            ]
            
            # Create output directories
            output_base = Path(f"outputs/{region_name}")
            executed_nb_dir = output_base / "executed_notebooks"
            executed_nb_dir.mkdir(parents=True, exist_ok=True)
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would process {len(notebooks)} notebooks for {region_name}")
                return True
            
            # Execute notebooks in sequence
            failed_notebooks = []
            
            for i, notebook_path in enumerate(notebooks, 1):
                notebook_name = Path(notebook_path).name
                output_name = notebook_name.replace('.ipynb', '_executed.ipynb')
                output_path = executed_nb_dir / output_name
                
                self.logger.info(f"Executing notebook {i}/{len(notebooks)}: {notebook_path}")
                
                try:
                    # Load region configuration
                    config_path = Path("config/location_config.json")
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                    
                    # Execute notebook with papermill
                    pm.execute_notebook(
                        input_path=notebook_path,
                        output_path=str(output_path),
                        parameters={
                            "region_name": region_name,
                            "output_dir": str(output_base)
                        },
                        kernel_name="python3"
                    )
                    
                    self.logger.info(f"Successfully completed: {notebook_path}")
                    
                except Exception as e:
                    error_str = str(e).encode('ascii', errors='ignore').decode('ascii')
                    self.logger.error(f"Failed to execute {notebook_name}: {error_str}")
                    failed_notebooks.append(notebook_name)
                    
                    # Continue with next notebook instead of stopping
                    continue
            
            processing_end = datetime.now()
            self.logger.info(f"Processing completed at: {processing_end}")
            
            if failed_notebooks:
                self.logger.warning(f"Region {region_name} completed with {len(failed_notebooks)} failed notebooks: {failed_notebooks}")
                self.logger.info("Status: PARTIAL SUCCESS")
                return False
            else:
                self.logger.info("Status: SUCCESS")
                return True
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed for {region_name}: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.logger.info("Status: FAILED")
            return False
    
    def display_progress(self, regions: List[Tuple[str, Path]], current_idx: int = None):
        """Display clean progress table"""
        print("\n" + "-" * 50)
        print("PROCESSING SAFE REGIONS")
        print("-" * 50)
        
        for i, (region_name, _) in enumerate(regions):
            idx_display = f"[{i+1}/{len(regions)}]"
            
            if current_idx is None:
                # Initial display
                if self.is_region_processed(region_name) and not self.force_reprocess:
                    status = "⏭ Skipped (already processed)"
                else:
                    status = "⏳ Pending"
            else:
                # Progress display
                if i < current_idx:
                    if region_name in self.results['processed']:
                        status = "✔ Completed"
                    elif region_name in self.results['failed']:
                        status = "❌ Failed"
                    else:
                        status = "⏭ Skipped"
                elif i == current_idx:
                    status = "⏳ Processing"
                else:
                    status = "⏳ Pending"
            
            print(f"{idx_display:<8} {region_name:<20} {status}")
        
        print("-" * 50)
    
    def process_all_regions(self, target_region: Optional[str] = None):
        """Main function to process all discovered regions"""
        # Discover regions
        regions = self.discover_regions()
        
        if not regions:
            print("❌ No regions with SAFE files found!")
            return
        
        # Filter to specific region if requested
        if target_region:
            regions = [(name, path) for name, path in regions if name.lower() == target_region.lower()]
            if not regions:
                print(f"❌ Region '{target_region}' not found or has no SAFE files!")
                return
        
        print(f"\n🔍 Found {len(regions)} regions with SAFE files:")
        for region_name, _ in regions:
            print(f"   - {region_name}")
        
        # Display initial progress
        self.display_progress(regions)
        
        if self.dry_run:
            print("\n[DRY RUN] No actual processing performed")
            return
        
        # Process each region
        for i, (region_name, region_dir) in enumerate(regions):
            self.display_progress(regions, i)
            
            # Check if already processed
            if self.is_region_processed(region_name) and not self.force_reprocess:
                print(f"\n[SKIP] Region '{region_name}' already processed.")
                self.results['skipped'].append(region_name)
                continue
            
            print(f"\n[RUN] Processing region '{region_name}'...")
            
            # Configure region
            if not self.configure_region(region_name, region_dir):
                print(f"[ERROR] Failed to configure region '{region_name}'. Continuing...")
                self.results['failed'].append(region_name)
                continue
            
            # Run pipeline
            success = self.run_pipeline_for_region(region_name)
            
            if success:
                print(f"[SUCCESS] Completed processing '{region_name}'")
                self.results['processed'].append(region_name)
            else:
                print(f"[ERROR] Processing failed for '{region_name}'. Continuing...")
                self.results['failed'].append(region_name)
        
        # Final progress display
        self.display_progress(regions, len(regions))
        
        # Print final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print comprehensive final summary"""
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        
        processed_count = len(self.results['processed'])
        skipped_count = len(self.results['skipped'])
        failed_count = len(self.results['failed'])
        
        print(f"Processed: {processed_count} regions")
        print(f"Skipped:   {skipped_count} regions")
        print(f"Failed:    {failed_count} regions")
        
        if self.results['processed']:
            print(f"\n✔ Successfully processed:")
            for region in self.results['processed']:
                print(f"   - {region}")
                print(f"     Output: outputs/{region}/final_showcase/")
        
        if self.results['skipped']:
            print(f"\n⏭ Skipped (already processed):")
            for region in self.results['skipped']:
                print(f"   - {region}")
        
        if self.results['failed']:
            print(f"\n❌ Failed:")
            for region in self.results['failed']:
                print(f"   - {region}")
        
        print(f"\n📋 Full log: {self.log_file}")
        print("=" * 40)
        
        # Log final summary
        self.logger.info("=== PROCESSING SUMMARY ===")
        self.logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}")
        self.logger.info("=== AUTO-PROCESS ALL REGIONS COMPLETED ===")

def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Auto-process all regions with SAFE files through the SDB pipeline"
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force reprocessing even if region already processed'
    )
    parser.add_argument(
        '--region', 
        type=str, 
        help='Process only the specified region'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be processed without actually running'
    )
    
    args = parser.parse_args()
    
    print("🛰️ AUTO-PROCESS ALL SAFE REGIONS")
    print("=" * 50)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual processing will occur")
    if args.force:
        print("🔄 FORCE MODE - Will reprocess existing regions")
    if args.region:
        print(f"🎯 TARGET MODE - Processing only region: {args.region}")
    
    try:
        processor = RegionProcessor(
            force_reprocess=args.force,
            dry_run=args.dry_run
        )
        processor.process_all_regions(args.region)
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()