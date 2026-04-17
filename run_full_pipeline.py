#!/usr/bin/env python3
"""
Full SDB Pipeline Runner
========================
Automated execution of the complete Satellite-Derived Bathymetry pipeline
using papermill for notebook orchestration.

This script runs all pipeline notebooks in sequence with proper parameter injection,
dynamically loads region configuration, and organizes all outputs systematically.

Usage:
    python run_full_pipeline.py

Author: SDB Pipeline Team
Date: November 12, 2025
"""

import papermill as pm
import os
import json
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(region_name):
    """Setup logging for pipeline execution"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"full_pipeline_execution_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_configuration():
    """Load region configuration from location_config.json"""
    try:
        config_path = Path("config/location_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file 'config/location_config.json' not found!")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in configuration file!")

def main():
    """Main pipeline execution function"""
    
    # Load configuration
    print("[CONFIG] Loading configuration...")
    config = load_configuration()
    region = config["region_name"]
    
    # Setup logging
    logger = setup_logging(region)
    logger.info(f"Starting full SDB pipeline for region: {region}")
    
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
    output_base = Path(f"outputs/{region}")
    executed_nb_dir = output_base / "executed_notebooks"
    executed_nb_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[PIPELINE] Starting full SDB pipeline for region: {region}")
    print(f"[OUTPUT] Output directory: {output_base}")
    print(f"[NOTEBOOKS] Executed notebooks will be saved to: {executed_nb_dir}")
    print("=" * 60)
    
    # Execute notebooks in sequence
    total_notebooks = len(notebooks)
    failed_notebooks = []
    
    for i, notebook_path in enumerate(notebooks, 1):
        notebook_name = Path(notebook_path).name
        output_name = notebook_name.replace('.ipynb', '_executed.ipynb')
        output_path = executed_nb_dir / output_name
        
        print(f"\n[{i}/{total_notebooks}] [RUNNING] {notebook_name}...")
        logger.info(f"Executing notebook {i}/{total_notebooks}: {notebook_path}")
        
        try:
            # Execute notebook with papermill
            pm.execute_notebook(
                input_path=notebook_path,
                output_path=str(output_path),
                parameters={
                    "region_name": region,
                    "safe_file_path": config.get("safe_file_path", ""),
                    "output_dir": str(output_base)
                },
                kernel_name="python3"
            )
            
            print(f"[SUCCESS] Completed {notebook_name}")
            logger.info(f"Successfully completed: {notebook_path}")
            
        except Exception as e:
            # Clean error message to avoid Unicode issues in logging
            error_str = str(e).encode('ascii', errors='ignore').decode('ascii')
            error_msg = f"[ERROR] Failed to execute {notebook_name}: {error_str}"
            print(error_msg)
            logger.error(f"Failed to execute {notebook_name}: {error_str}")
            failed_notebooks.append(notebook_name)
            
            # Ask user if they want to continue
            response = input(f"\nNotebook {notebook_name} failed. Continue with remaining notebooks? (y/n): ")
            if response.lower() != 'y':
                logger.info("Pipeline execution stopped by user after failure")
                break
    
    # Final summary
    print("\n" + "=" * 60)
    if failed_notebooks:
        print(f"[WARNING] Pipeline completed with {len(failed_notebooks)} failures:")
        for failed_nb in failed_notebooks:
            print(f"   - {failed_nb}")
        logger.warning(f"Pipeline completed with failures: {failed_notebooks}")
    else:
        print("[SUCCESS] Pipeline completed successfully!")
        logger.info("Full pipeline execution completed successfully")
    
    print(f"\n[RESULTS] Results available in: outputs/{region}/")
    print(f"[VISUALIZATIONS] Final visualizations: outputs/{region}/final_showcase/")
    print(f"[NOTEBOOKS] Executed notebooks: outputs/{region}/executed_notebooks/")
    print(f"[LOGS] Execution logs: logs/full_pipeline_execution_*.log")
    
    # Display key output files
    showcase_dir = output_base / "final_showcase"
    if showcase_dir.exists():
        print(f"\n[OUTPUTS] Key outputs generated:")
        for file_path in showcase_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.html', '.png', '.csv', '.json']:
                print(f"   - {file_path.relative_to(output_base)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING] Pipeline execution interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with error: {e}")
        logging.error(f"Pipeline failed: {e}")
        exit(1)