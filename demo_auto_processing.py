#!/usr/bin/env python3
"""
Demo Script - Show Auto-Processing Capabilities
==============================================
Demonstrates the auto-processing system functionality
"""

import subprocess
import sys
from pathlib import Path

def run_demo():
    """Run demonstration of auto-processing capabilities"""
    
    print("🎯 SAFE FILE AUTO-PROCESSING SYSTEM DEMO")
    print("=" * 60)
    print()
    
    print("This system can automatically process all your regions with SAFE files!")
    print()
    
    # Show available regions
    print("📂 AVAILABLE REGIONS WITH SAFE FILES:")
    result = subprocess.run([
        sys.executable, "process_all_regions.py", "--dry-run"
    ], capture_output=True, text=True)
    
    if "Found" in result.stdout:
        lines = result.stdout.split('\n')
        for line in lines:
            if line.strip().startswith('- '):
                region_name = line.strip()[2:]
                # Get SAFE file info
                safe_files = []
                data_path = Path(f"data/sentinel/{region_name}/raw")
                if data_path.exists():
                    for file in data_path.iterdir():
                        if file.name.endswith('.zip') and 'S2' in file.name:
                            safe_files.append(file.name)
                
                print(f"   🌍 {region_name.upper()}")
                for safe_file in safe_files:
                    print(f"      📁 {safe_file}")
        print()
    
    print("🚀 USAGE EXAMPLES:")
    print()
    print("# Process all regions automatically:")
    print("python process_all_regions.py")
    print()
    print("# Process only a specific region:")
    print("python process_all_regions.py --region goa")
    print()
    print("# Show what would be processed (dry run):")
    print("python process_all_regions.py --dry-run")
    print()
    print("# Force reprocessing of completed regions:")
    print("python process_all_regions.py --force")
    print()
    
    print("✨ BENEFITS:")
    print("   ✅ Automatic SAFE file detection")
    print("   ✅ Intelligent region-to-coordinates mapping")
    print("   ✅ Skip already processed regions")
    print("   ✅ Comprehensive error handling")
    print("   ✅ Detailed progress tracking")
    print("   ✅ Region-specific output organization")
    print()
    
    print("📁 OUTPUT STRUCTURE:")
    print("   outputs/")
    print("   ├── goa/final_showcase/")
    print("   ├── kachchh/final_showcase/")
    print("   └── palk_strait/final_showcase/")
    print()
    
    print("🎉 Ready to process all your regions automatically!")

if __name__ == "__main__":
    run_demo()