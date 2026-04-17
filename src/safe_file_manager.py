#!/usr/bin/env python3
"""
SAFE File Manager for Automatic Region Configuration
Automatically selects and configures the correct SAFE file based on location_config.json
"""

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

class SAFEFileManager:
    """Manages SAFE files for different regions and coordinates"""
    
    def __init__(self, config_path: str = "config/location_config.json"):
        self.config_path = Path(config_path)
        self.data_root = Path("data/sentinel")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load current location configuration
        self.location_config = self.load_location_config()
    
    def load_location_config(self) -> Dict:
        """Load the current location configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Location config not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in location config: {e}")
            return {}
    
    def scan_safe_files(self) -> Dict[str, List[str]]:
        """Scan all regions for SAFE files"""
        safe_files_map = {}
        
        if not self.data_root.exists():
            self.logger.error(f"Data root directory not found: {self.data_root}")
            return safe_files_map
        
        for region_dir in self.data_root.iterdir():
            if region_dir.is_dir():
                region_name = region_dir.name
                raw_dir = region_dir / "raw"
                
                if raw_dir.exists():
                    safe_files = []
                    
                    for item in raw_dir.iterdir():
                        if item.is_file() and item.name.endswith('.zip'):
                            if self.is_safe_zip(item):
                                safe_files.append(item.name)
                        elif item.is_dir() and item.name.endswith('.SAFE'):
                            safe_files.append(item.name)
                    
                    if safe_files:
                        safe_files_map[region_name] = sorted(safe_files)
                        self.logger.info(f"Found {len(safe_files)} SAFE files in {region_name}")
        
        return safe_files_map
    
    def is_safe_zip(self, zip_path: Path) -> bool:
        """Check if a zip file contains Sentinel-2 SAFE data"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                files = zf.namelist()
                # Check for typical SAFE structure
                return any(
                    f.endswith('.SAFE/') or 
                    'MTD_MSIL2A.xml' in f or 
                    'MTD_MSIL1C.xml' in f or
                    'GRANULE/' in f
                    for f in files
                )
        except:
            return False
    
    def parse_safe_filename(self, filename: str) -> Dict:
        """Parse Sentinel-2 SAFE filename to extract metadata"""
        # Sentinel-2 filename format:
        # S2A_MSIL2A_20251106T053241_N0511_R105_T43PCS_20251106T083512.SAFE.zip
        
        metadata = {
            'satellite': 'Unknown',
            'processing_level': 'Unknown',
            'sensing_date': 'Unknown',
            'pdgs_version': 'Unknown',
            'relative_orbit': 'Unknown',
            'tile_id': 'Unknown',
            'product_date': 'Unknown',
            'cloud_coverage': 'Unknown'
        }
        
        # Remove .SAFE.zip or .SAFE extension
        clean_name = filename.replace('.SAFE.zip', '').replace('.SAFE', '')
        
        # Parse using regex
        pattern = r'S2([ABC])_MSIL([12][AC])_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\w{5})_(\d{8}T\d{6})'
        match = re.match(pattern, clean_name)
        
        if match:
            metadata['satellite'] = f"Sentinel-2{match.group(1)}"
            metadata['processing_level'] = f"Level-{match.group(2)}"
            metadata['sensing_date'] = match.group(3)
            metadata['pdgs_version'] = match.group(4)
            metadata['relative_orbit'] = match.group(5)
            metadata['tile_id'] = match.group(6)
            metadata['product_date'] = match.group(7)
        
        return metadata
    
    def score_safe_file(self, filename: str, region: str) -> float:
        """Score a SAFE file based on various quality criteria"""
        score = 0.0
        metadata = self.parse_safe_filename(filename)
        
        # Processing level preference (Level-2A is preferred)
        if 'Level-2A' in metadata['processing_level']:
            score += 20
        elif 'Level-1C' in metadata['processing_level']:
            score += 10
        
        # Date preference (more recent is better)
        try:
            sensing_date = datetime.strptime(metadata['sensing_date'], '%Y%m%dT%H%M%S')
            days_ago = (datetime.now() - sensing_date).days
            if days_ago <= 30:
                score += 15
            elif days_ago <= 90:
                score += 10
            elif days_ago <= 180:
                score += 5
        except:
            pass
        
        # Satellite preference (newer satellites might be preferred)
        if 'Sentinel-2A' in metadata['satellite']:
            score += 5
        elif 'Sentinel-2B' in metadata['satellite']:
            score += 6
        elif 'Sentinel-2C' in metadata['satellite']:
            score += 7
        
        return score
    
    def find_region_for_coordinates(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Optional[str]:
        """Find the best matching region based on coordinates"""
        # Define approximate coordinate ranges for each region
        region_coords = {
            'goa': {'lat': (15.0, 15.8), 'lon': (73.7, 74.3)},
            'kachchh': {'lat': (22.5, 24.0), 'lon': (68.0, 71.0)},
            'palk_strait': {'lat': (8.5, 10.0), 'lon': (78.5, 80.5)},
            'lakshadweep': {'lat': (10.5, 12.5), 'lon': (71.5, 74.0)},
            'andaman': {'lat': (11.5, 13.5), 'lon': (92.0, 94.0)},
        }
        
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        
        best_match = None
        min_distance = float('inf')
        
        for region, coords in region_coords.items():
            region_lat_center = (coords['lat'][0] + coords['lat'][1]) / 2
            region_lon_center = (coords['lon'][0] + coords['lon'][1]) / 2
            
            distance = ((lat_center - region_lat_center) ** 2 + (lon_center - region_lon_center) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_match = region
        
        return best_match
    
    def get_region_from_config(self) -> str:
        """Get region name from current configuration"""
        region_name = self.location_config.get('region_name', '').lower()
        
        # Handle region name mapping
        region_mapping = {
            'rameswaram_palkstrait': 'palk_strait',
            'rameswaram': 'palk_strait',
            'palk strait': 'palk_strait',
            'palk_strait': 'palk_strait',
            'goa': 'goa',
            'kachchh': 'kachchh',
            'kutch': 'kachchh',
            'lakshadweep': 'lakshadweep',
            'andaman': 'andaman',
            'andaman islands': 'andaman'
        }
        
        # Try direct mapping first
        for key, value in region_mapping.items():
            if key in region_name or region_name in key:
                return value
        
        # If no mapping found, try coordinate-based detection
        aoi = self.location_config.get('aoi', {})
        if all(key in aoi for key in ['min_lat', 'max_lat', 'min_lon', 'max_lon']):
            detected_region = self.find_region_for_coordinates(
                aoi['min_lat'], aoi['max_lat'], aoi['min_lon'], aoi['max_lon']
            )
            if detected_region:
                self.logger.info(f"Detected region from coordinates: {detected_region}")
                return detected_region
        
        return region_name
    
    def get_best_safe_file(self) -> Optional[Path]:
        """Get the best SAFE file for the current region"""
        region = self.get_region_from_config()
        safe_files_map = self.scan_safe_files()
        
        if region not in safe_files_map:
            self.logger.warning(f"No SAFE files found for region: {region}")
            return None
        
        files = safe_files_map[region]
        if not files:
            return None
        
        if len(files) == 1:
            safe_path = self.data_root / region / "raw" / files[0]
            self.logger.info(f"Using only available SAFE file: {files[0]}")
            return safe_path
        
        # Score and select best file
        scored_files = [(f, self.score_safe_file(f, region)) for f in files]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        best_file = scored_files[0][0]
        safe_path = self.data_root / region / "raw" / best_file
        
        self.logger.info(f"Selected best SAFE file: {best_file} (score: {scored_files[0][1]:.1f})")
        return safe_path
    
    def extract_safe_metadata(self, safe_path: Path) -> Dict:
        """Extract comprehensive metadata from SAFE file"""
        metadata = self.parse_safe_filename(safe_path.name)
        
        # Add file system information
        metadata.update({
            'path': str(safe_path),
            'name': safe_path.name,
            'exists': safe_path.exists(),
            'size_mb': 0
        })
        
        if safe_path.exists():
            try:
                if safe_path.is_file():
                    metadata['size_mb'] = safe_path.stat().st_size / (1024 * 1024)
                elif safe_path.is_dir():
                    total_size = sum(f.stat().st_size for f in safe_path.rglob('*') if f.is_file())
                    metadata['size_mb'] = total_size / (1024 * 1024)
            except:
                pass
        
        return metadata
    
    def generate_report(self) -> str:
        """Generate comprehensive SAFE files inventory report"""
        report = []
        report.append("=" * 70)
        report.append("SENTINEL-2 SAFE FILES INVENTORY REPORT")
        report.append("=" * 70)
        report.append("")
        
        safe_files_map = self.scan_safe_files()
        
        if not safe_files_map:
            report.append("X No SAFE files found!")
            report.append("")
            report.append("Expected structure:")
            report.append("  data/sentinel/{region}/raw/*.SAFE.zip")
            return "\n".join(report)
        
        total_files = sum(len(files) for files in safe_files_map.values())
        report.append(f"CHART Total regions with SAFE files: {len(safe_files_map)}")
        report.append(f"CHART Total SAFE files found: {total_files}")
        report.append("")
        
        for region, files in safe_files_map.items():
            report.append(f"WORLD Region: {region.upper()}")
            report.append("-" * 50)
            
            for i, safe_file in enumerate(files, 1):
                safe_path = self.data_root / region / "raw" / safe_file
                metadata = self.extract_safe_metadata(safe_path)
                score = self.score_safe_file(safe_file, region)
                
                report.append(f"  [{i}] {safe_file}")
                report.append(f"      SATELLITE  Satellite: {metadata['satellite']}")
                report.append(f"      CALENDAR Date: {metadata['sensing_date']}")
                report.append(f"      MICROSCOPE Level: {metadata['processing_level']}")
                report.append(f"      TARGET Tile: {metadata['tile_id']}")
                report.append(f"      DISK Size: {metadata['size_mb']:.1f} MB")
                report.append(f"      STAR Score: {score:.1f}")
                report.append("")
            
            # Show recommended file
            if files:
                scored = [(f, self.score_safe_file(f, region)) for f in files]
                best = max(scored, key=lambda x: x[1])
                report.append(f"    TROPHY RECOMMENDED: {best[0]} (score: {best[1]:.1f})")
            report.append("")
        
        # Current region analysis
        current_region = self.get_region_from_config()
        report.append(f"TARGET CURRENT CONFIGURATION:")
        report.append(f"   Region name: {self.location_config.get('region_name', 'Not set')}")
        report.append(f"   Detected region: {current_region}")
        
        best_file = self.get_best_safe_file()
        if best_file:
            report.append(f"   Selected SAFE file: {best_file.name}")
            report.append(f"   Full path: {best_file}")
        else:
            report.append(f"   X No SAFE file available for region: {current_region}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

def main():
    """Main function to test SAFE file manager"""
    print("🔍 SAFE FILE SCANNER")
    print("=" * 50)
    
    manager = SAFEFileManager()
    print(manager.generate_report())

if __name__ == "__main__":
    main()