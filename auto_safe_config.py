#!/usr/bin/env python3
"""
Automatic SAFE Configuration System
Monitors location_config.json changes and automatically configures the correct SAFE file
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.safe_file_manager import SAFEFileManager
from src.band_extraction_config import BandExtractionConfig

class AutoSAFEConfigurator:
    """Automatically configures SAFE files based on location changes"""
    
    def __init__(self):
        self.config_path = Path("config/location_config.json")
        self.safe_manager = SAFEFileManager(str(self.config_path))
        self.band_config = BandExtractionConfig(self.safe_manager)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Track last configuration state
        self.last_config = {}
        self.last_modified = 0
    
    def check_config_changes(self) -> bool:
        """Check if location_config.json has been modified"""
        if not self.config_path.exists():
            return False
        
        current_modified = self.config_path.stat().st_mtime
        
        if current_modified != self.last_modified:
            self.last_modified = current_modified
            return True
        
        return False
    
    def load_current_config(self) -> Dict:
        """Load current location configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def has_config_changed(self, current_config: Dict) -> bool:
        """Check if configuration content has actually changed"""
        # Compare key fields that would affect SAFE file selection
        key_fields = ['region_name', 'aoi']
        
        for field in key_fields:
            if self.last_config.get(field) != current_config.get(field):
                return True
        
        return False
    
    def configure_for_region(self, config: Dict) -> bool:
        """Configure SAFE file and band extraction for the current region"""
        self.logger.info("🔧 Configuring SAFE file for current region...")
        
        # Reload SAFE manager with new config
        self.safe_manager = SAFEFileManager(str(self.config_path))
        region = self.safe_manager.get_region_from_config()
        
        if not region:
            self.logger.error("❌ Could not determine region from configuration")
            return False
        
        # Find best SAFE file
        best_safe = self.safe_manager.get_best_safe_file()
        
        if not best_safe:
            self.logger.error(f"❌ No SAFE files available for region: {region}")
            return False
        
        self.logger.info(f"✅ Selected SAFE file: {best_safe.name}")
        
        # Generate band extraction configuration
        self.band_config = BandExtractionConfig(self.safe_manager)
        config_path = self.band_config.create_config_file()
        
        if not config_path:
            self.logger.error("❌ Failed to create band extraction configuration")
            return False
        
        self.logger.info(f"✅ Band extraction config created: {config_path}")
        
        # Validate SAFE structure
        valid = self.band_config.validate_safe_structure()
        
        if not valid:
            self.logger.warning("⚠️ SAFE file structure validation failed - proceeding anyway")
        else:
            self.logger.info("✅ SAFE file structure validated")
        
        # Update notebook configurations if needed
        self._update_notebook_configs(region, best_safe)
        
        return True
    
    def _update_notebook_configs(self, region: str, safe_path: Path) -> None:
        """Update notebook configuration files with new SAFE file paths"""
        # Update config files that notebooks might use
        notebook_config = {
            'region': region,
            'safe_file_path': str(safe_path),
            'safe_file_name': safe_path.name,
            'data_root': f"data/sentinel/{region}",
            'output_root': f"outputs/{region}",
            'processing_level': self.safe_manager.extract_safe_metadata(safe_path).get('processing_level', 'Unknown'),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save notebook-specific config
        notebook_config_path = Path("config/notebook_config.json")
        notebook_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(notebook_config_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Notebook configuration updated: {notebook_config_path}")
    
    def run_auto_configuration(self) -> None:
        """Run automatic configuration when location_config.json changes"""
        current_config = self.load_current_config()
        
        if not current_config:
            self.logger.error("❌ Could not load current configuration")
            return
        
        if not self.has_config_changed(current_config):
            self.logger.debug("No significant configuration changes detected")
            return
        
        self.logger.info("📍 Location configuration changed - reconfiguring SAFE files...")
        
        # Store new config
        self.last_config = current_config.copy()
        
        # Configure for new region
        success = self.configure_for_region(current_config)
        
        if success:
            self.logger.info("🎉 SAFE file configuration completed successfully!")
            self._print_configuration_summary()
        else:
            self.logger.error("❌ SAFE file configuration failed!")
    
    def _print_configuration_summary(self) -> None:
        """Print a summary of the current configuration"""
        region = self.safe_manager.get_region_from_config()
        best_safe = self.safe_manager.get_best_safe_file()
        
        if not best_safe:
            return
        
        metadata = self.safe_manager.extract_safe_metadata(best_safe)
        
        print("\n" + "="*60)
        print("SATELLITE CURRENT SAFE FILE CONFIGURATION")
        print("="*60)
        print(f"WORLD Region: {region.upper()}")
        print(f"FOLDER SAFE file: {best_safe.name}")
        print(f"SATELLITE Satellite: {metadata.get('satellite', 'Unknown')}")
        print(f"CALENDAR Date: {metadata.get('sensing_date', 'Unknown')}")
        print(f"MICROSCOPE Level: {metadata.get('processing_level', 'Unknown')}")
        print(f"TARGET Tile: {metadata.get('tile_id', 'Unknown')}")
        print(f"DISK Size: {metadata.get('size_mb', 0):.1f} MB")
        print(f"FOLDER Path: {best_safe}")
        print("="*60)
        print()
    
    def monitor_config_file(self, interval: int = 2) -> None:
        """Monitor location_config.json for changes and auto-configure"""
        self.logger.info(f"🔍 Monitoring {self.config_path} for changes...")
        self.logger.info(f"⏱️ Check interval: {interval} seconds")
        self.logger.info("Press Ctrl+C to stop monitoring")
        
        # Initial configuration
        self.run_auto_configuration()
        
        try:
            while True:
                time.sleep(interval)
                
                if self.check_config_changes():
                    self.logger.info("📝 Configuration file modified - checking for changes...")
                    self.run_auto_configuration()
                    
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Monitoring stopped by user")
    
    def run_single_configuration(self) -> bool:
        """Run configuration once without monitoring"""
        self.logger.info("🔧 Running single SAFE file configuration...")
        
        current_config = self.load_current_config()
        if not current_config:
            return False
        
        success = self.configure_for_region(current_config)
        
        if success:
            self._print_configuration_summary()
        
        return success

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automatic SAFE File Configuration System")
    parser.add_argument('--monitor', action='store_true', help='Monitor config file for changes')
    parser.add_argument('--interval', type=int, default=2, help='Monitoring interval in seconds (default: 2)')
    parser.add_argument('--report', action='store_true', help='Show SAFE files inventory report')
    
    args = parser.parse_args()
    
    configurator = AutoSAFEConfigurator()
    
    if args.report:
        # Show inventory report
        print(configurator.safe_manager.generate_report())
        return
    
    if args.monitor:
        # Monitor mode
        configurator.monitor_config_file(args.interval)
    else:
        # Single run mode
        success = configurator.run_single_configuration()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()