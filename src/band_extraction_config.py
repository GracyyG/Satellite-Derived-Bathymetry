#!/usr/bin/env python3
"""
Band Extraction Configuration Manager
Automatically configures band extraction based on selected SAFE files
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.safe_file_manager import SAFEFileManager

class BandExtractionConfig:
    """Manages band extraction configuration based on SAFE files"""
    
    def __init__(self, safe_manager: Optional[SAFEFileManager] = None):
        self.safe_manager = safe_manager or SAFEFileManager()
        self.logger = logging.getLogger(__name__)
        
        # Band configuration for different processing levels
        self.band_configs = {
            'Level-2A': {
                'bands': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
                'resolution': {
                    'B02': '10m', 'B03': '10m', 'B04': '10m', 'B08': '10m',  # 10m bands
                    'B05': '20m', 'B06': '20m', 'B07': '20m', 'B8A': '20m', 'B11': '20m', 'B12': '20m'  # 20m bands
                },
                'folder_structure': {
                    '10m': 'GRANULE/*/IMG_DATA/R10m/',
                    '20m': 'GRANULE/*/IMG_DATA/R20m/',
                    '60m': 'GRANULE/*/IMG_DATA/R60m/'
                },
                'file_pattern': {
                    'B02': '*_B02_10m.jp2',
                    'B03': '*_B03_10m.jp2',
                    'B04': '*_B04_10m.jp2',
                    'B08': '*_B08_10m.jp2',
                    'B05': '*_B05_20m.jp2',
                    'B06': '*_B06_20m.jp2',
                    'B07': '*_B07_20m.jp2',
                    'B8A': '*_B8A_20m.jp2',
                    'B11': '*_B11_20m.jp2',
                    'B12': '*_B12_20m.jp2'
                }
            },
            'Level-1C': {
                'bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
                'resolution': {
                    'B02': '10m', 'B03': '10m', 'B04': '10m', 'B08': '10m',  # 10m bands
                    'B05': '20m', 'B06': '20m', 'B07': '20m', 'B8A': '20m', 'B11': '20m', 'B12': '20m',  # 20m bands
                    'B01': '60m', 'B09': '60m', 'B10': '60m'  # 60m bands
                },
                'folder_structure': {
                    '10m': 'GRANULE/*/IMG_DATA/',
                    '20m': 'GRANULE/*/IMG_DATA/',
                    '60m': 'GRANULE/*/IMG_DATA/'
                },
                'file_pattern': {
                    'B01': '*_B01.jp2',
                    'B02': '*_B02.jp2',
                    'B03': '*_B03.jp2',
                    'B04': '*_B04.jp2',
                    'B05': '*_B05.jp2',
                    'B06': '*_B06.jp2',
                    'B07': '*_B07.jp2',
                    'B08': '*_B08.jp2',
                    'B8A': '*_B8A.jp2',
                    'B09': '*_B09.jp2',
                    'B10': '*_B10.jp2',
                    'B11': '*_B11.jp2',
                    'B12': '*_B12.jp2'
                }
            }
        }
    
    def get_band_extraction_config(self) -> Dict:
        """Generate band extraction configuration for current region"""
        best_safe = self.safe_manager.get_best_safe_file()
        
        if not best_safe:
            self.logger.error("No SAFE file available for band extraction configuration")
            return {}
        
        metadata = self.safe_manager.extract_safe_metadata(best_safe)
        processing_level = metadata.get('processing_level', 'Unknown')
        region = self.safe_manager.get_region_from_config()
        
        if processing_level not in self.band_configs:
            self.logger.error(f"Unsupported processing level: {processing_level}")
            return {}
        
        config = self.band_configs[processing_level].copy()
        
        # Add SAFE file specific information
        config['safe_file'] = {
            'path': str(best_safe),
            'name': best_safe.name,
            'processing_level': processing_level,
            'satellite': metadata.get('satellite', 'Unknown'),
            'sensing_date': metadata.get('sensing_date', 'Unknown'),
            'tile_id': metadata.get('tile_id', 'Unknown')
        }
        
        config['region'] = region
        config['output_paths'] = self._generate_output_paths(region)
        
        return config
    
    def _generate_output_paths(self, region: str) -> Dict:
        """Generate output paths for processed data"""
        base_path = Path(f"data/sentinel/{region}")
        
        return {
            'processed': str(base_path / "processed"),
            'bands': str(base_path / "processed" / "bands"),
            'rgb': str(base_path / "processed" / "rgb"),
            'indices': str(base_path / "processed" / "indices"),
            'metadata': str(base_path / "processed" / "metadata"),
            'thumbnails': str(base_path / "processed" / "thumbnails")
        }
    
    def create_config_file(self, output_path: str = "config/band_extraction_config.json") -> str:
        """Create a configuration file for band extraction"""
        config = self.get_band_extraction_config()
        
        if not config:
            return ""
        
        config_path = Path(output_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp and generation info
        from datetime import datetime
        config['_metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'generated_by': 'BandExtractionConfig',
            'region': config.get('region'),
            'safe_file': config.get('safe_file', {}).get('name')
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Band extraction config saved to: {config_path}")
        return str(config_path)
    
    def validate_safe_structure(self) -> bool:
        """Validate that the selected SAFE file has the expected structure"""
        best_safe = self.safe_manager.get_best_safe_file()
        
        if not best_safe or not best_safe.exists():
            self.logger.error(f"SAFE file not found: {best_safe}")
            return False
        
        metadata = self.safe_manager.extract_safe_metadata(best_safe)
        processing_level = metadata.get('processing_level', 'Unknown')
        
        if processing_level not in self.band_configs:
            self.logger.error(f"Unsupported processing level: {processing_level}")
            return False
        
        # For .zip files, we'll validate during extraction
        if best_safe.name.endswith('.zip'):
            self.logger.info(f"ZIP SAFE file validation will be done during extraction: {best_safe.name}")
            return True
        
        # For unzipped .SAFE directories
        if best_safe.is_dir():
            config = self.band_configs[processing_level]
            missing_bands = []
            
            for band in config['bands'][:5]:  # Check first 5 bands as sample
                resolution = config['resolution'].get(band, '10m')
                folder_pattern = config['folder_structure'][resolution]
                
                # This is a simplified check - would need more sophisticated validation
                if not list(best_safe.rglob(f'*{band}*.jp2')):
                    missing_bands.append(band)
            
            if missing_bands:
                self.logger.warning(f"Some bands may be missing: {missing_bands[:3]}...")
            
            return len(missing_bands) < len(config['bands']) / 2  # Allow some missing bands
        
        return False
    
    def get_extraction_command(self) -> str:
        """Generate the command to extract bands from SAFE file"""
        config = self.get_band_extraction_config()
        
        if not config:
            return ""
        
        safe_info = config.get('safe_file', {})
        safe_path = safe_info.get('path', '')
        region = config.get('region', '')
        
        if not safe_path or not region:
            return ""
        
        # Generate Python command for band extraction
        command_parts = [
            "python",
            "scripts/extract_sentinel_bands.py",
            f"--safe-file '{safe_path}'",
            f"--region '{region}'",
            f"--output-dir 'data/sentinel/{region}/processed'",
            f"--processing-level '{safe_info.get('processing_level', 'Level-2A')}'"
        ]
        
        return " ".join(command_parts)

def main():
    """Test band extraction configuration"""
    print("🛰️ BAND EXTRACTION CONFIGURATION")
    print("=" * 50)
    
    config_manager = BandExtractionConfig()
    
    # Generate configuration
    config = config_manager.get_band_extraction_config()
    
    if config:
        print(f"✅ Configuration generated for region: {config.get('region')}")
        print(f"📁 SAFE file: {config.get('safe_file', {}).get('name')}")
        print(f"🔬 Processing level: {config.get('safe_file', {}).get('processing_level')}")
        print(f"🎯 Available bands: {len(config.get('bands', []))}")
        
        # Save configuration
        config_path = config_manager.create_config_file()
        print(f"💾 Config saved to: {config_path}")
        
        # Validate structure
        valid = config_manager.validate_safe_structure()
        print(f"✔️ SAFE structure valid: {valid}")
        
        # Show extraction command
        command = config_manager.get_extraction_command()
        if command:
            print(f"🚀 Extraction command:")
            print(f"   {command}")
    else:
        print("❌ Could not generate configuration")

if __name__ == "__main__":
    main()