#!/usr/bin/env python3
"""
Demo script for the SDB Region Selector

This script demonstrates all the functionality of the region selector:
1. CLI interface
2. Configuration management  
3. Pipeline integration
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from run_region_selector import load_current_config, update_config, run_pipeline

def demo_cli_workflow():
    """Demonstrate the CLI workflow"""
    print("🌊 SDB Region Selector Demo")
    print("=" * 50)
    
    print("\n1. 📋 Current Configuration:")
    config = load_current_config()
    print(f"   Region: {config['region_name']}")
    print(f"   Bounds: {config['aoi']}")
    
    print("\n2. 🔄 Testing Configuration Update:")
    demo_regions = [
        {
            "name": "Maldives Demo",
            "bounds": {"min_lat": 3.0, "max_lat": 7.0, "min_lon": 72.0, "max_lon": 74.0}
        },
        {
            "name": "Andaman Demo", 
            "bounds": {"min_lat": 11.5, "max_lat": 13.5, "min_lon": 92.0, "max_lon": 94.0}
        }
    ]
    
    original_config = config.copy()
    
    for i, region in enumerate(demo_regions, 1):
        print(f"\n   Demo {i}: Updating to {region['name']}")
        success = update_config(
            region['name'],
            region['bounds']['min_lat'],
            region['bounds']['max_lat'], 
            region['bounds']['min_lon'],
            region['bounds']['max_lon']
        )
        
        if success:
            print(f"   ✅ Successfully updated to {region['name']}")
            # Verify update
            new_config = load_current_config()
            print(f"   📍 New region: {new_config['region_name']}")
        else:
            print(f"   ❌ Failed to update to {region['name']}")
    
    print("\n3. 🔙 Restoring Original Configuration:")
    restore_success = update_config(
        original_config['region_name'],
        original_config['aoi']['min_lat'],
        original_config['aoi']['max_lat'],
        original_config['aoi']['min_lon'],
        original_config['aoi']['max_lon']
    )
    
    if restore_success:
        print(f"   ✅ Original configuration restored: {original_config['region_name']}")
    else:
        print(f"   ❌ Failed to restore original configuration")
    
    print("\n4. 🚀 Available Interfaces:")
    print("   • CLI Mode:        python run_region_selector.py --cli")
    print("   • Streamlit Web:   python -m streamlit run run_region_selector.py")
    print("   • Quick Launch:    python launch_streamlit.py")
    
    print("\n5. 📁 Generated Files:")
    files_to_check = [
        "config/location_config.json",
        "logs/region_run_log.txt",
        "REGION_SELECTOR_README.md"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path} (exists)")
        else:
            print(f"   ❌ {file_path} (missing)")
    
    print("\n6. 🧪 Integration Test:")
    print("   Note: Pipeline execution test skipped to avoid long runtime")
    print("   To test pipeline: Choose 'y' when prompted in CLI mode")
    
    print(f"\n✨ Demo completed! The region selector is ready for use.")
    print(f"📖 See REGION_SELECTOR_README.md for full documentation.")

if __name__ == "__main__":
    demo_cli_workflow()