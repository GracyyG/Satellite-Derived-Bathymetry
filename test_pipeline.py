#!/usr/bin/env python3
"""
SDB Pipeline Test Script

Performs pre-flight checks before running the main pipeline:
1. Verifies configuration and directories
2. Checks for required notebooks
3. Validates data availability
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging

class PipelineTester:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.required_dirs = [
            'data',
            'outputs',
            'logs',
            'models',
            'data/sentinel',
            'data/gebco_reference',
            'data/processed'
        ]
        self.required_notebooks = [
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
        
        # Load configuration
        config_path = project_root / 'config' / 'location_config.json'
        try:
            with open(config_path) as f:
                self.config = json.load(f)
            self.region = self.config['region_name']
            self.region_slug = self.region.lower().replace(' ', '_')
        except FileNotFoundError:
            print("❌ Configuration file not found: config/location_config.json")
            sys.exit(1)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in configuration file")
            sys.exit(1)
            
    def print_header(self):
        """Print test header with region info"""
        print("\n" + "="*60)
        print(f"🌍 Testing pipeline for region: {self.region}")
        print("="*60 + "\n")
        
    def check_directories(self) -> List[str]:
        """Check if required directories exist, create if missing"""
        missing_dirs = []
        print("📁 Checking required directories...")
        
        for dir_name in self.required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                print(f"  Creating: {dir_name}/")
                dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_name}/")
            
        return missing_dirs
    
    def check_notebooks(self) -> List[str]:
        """Verify all required notebooks exist"""
        print("\n📔 Checking required notebooks...")
        missing_notebooks = []
        notebooks_dir = self.project_root / 'notebooks'
        
        for notebook in self.required_notebooks:
            notebook_path = notebooks_dir / notebook
            if not notebook_path.exists():
                missing_notebooks.append(notebook)
                print(f"  ❌ Missing: {notebook}")
            else:
                print(f"  ✓ {notebook}")
                
        return missing_notebooks
    
    def check_data(self) -> Dict[str, bool]:
        """Check for required input data"""
        print("\n💾 Checking data availability...")
        data_status = {
            'sentinel': False,
            'gebco': False,
            'icesat': False
        }
        
        # Check Sentinel-2 data
        sentinel_dir = self.project_root / 'data' / 'sentinel' / self.region_slug
        if sentinel_dir.exists() and any(sentinel_dir.iterdir()):
            data_status['sentinel'] = True
            print("  ✓ Sentinel-2 data found")
        else:
            print("  ⚠️  Sentinel-2 data missing")
            print("     Download using: python scripts/download_sentinel_aws.py")
            
        # Check GEBCO data
        gebco_dir = self.project_root / 'data' / 'gebco_reference'
        if gebco_dir.exists() and any(gebco_dir.iterdir()):
            data_status['gebco'] = True
            print("  ✓ GEBCO bathymetry data found")
        else:
            print("  ⚠️  GEBCO bathymetry data missing")
            print("     Download using: python scripts/download_gebco.py")
            
        # Check ICESat-2 data
        icesat_dir = self.project_root / 'data' / 'icesat2'
        if icesat_dir.exists() and any(icesat_dir.iterdir()):
            data_status['icesat'] = True
            print("  ✓ ICESat-2 data found")
        else:
            print("  ⚠️  ICESat-2 data missing")
            print("     Will be downloaded during pipeline execution")
            
        return data_status
    
    def run_tests(self) -> bool:
        """Run all pipeline tests"""
        self.print_header()
        
        # Check directories
        missing_dirs = self.check_directories()
        
        # Check notebooks
        missing_notebooks = self.check_notebooks()
        if missing_notebooks:
            print("\n❌ Error: Missing required notebooks")
            return False
            
        # Check data
        data_status = self.check_data()
        
        # Print summary
        print("\n📊 Test Summary:")
        print("  ✓ Configuration valid")
        print("  ✓ Directory structure ready")
        print("  ✓ All notebooks present")
        print(f"  {'✓' if data_status['sentinel'] else '⚠️'} Sentinel-2 data")
        print(f"  {'✓' if data_status['gebco'] else '⚠️'} GEBCO data")
        print(f"  {'✓' if data_status['icesat'] else '⚠️'} ICESat-2 data")
        
        # All critical checks passed
        print("\n✅ Pipeline ready to run!")
        print(f"   Region: {self.region}")
        print(f"   AOI: {self.config['aoi']['min_lat']}°N to {self.config['aoi']['max_lat']}°N, "
              f"{self.config['aoi']['min_lon']}°E to {self.config['aoi']['max_lon']}°E")
        print("\nRun pipeline with: python main_pipeline.py")
        
        return True

def main():
    """Main entry point"""
    try:
        # Get project root
        project_root = Path(__file__).parent
        
        # Run tests
        tester = PipelineTester(project_root)
        success = tester.run_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()