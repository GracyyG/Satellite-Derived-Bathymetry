#!/usr/bin/env python3
"""
Test the region selector functionality
"""

import json
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the functions we want to test
from run_region_selector import load_current_config, update_config

def test_config_operations():
    """Test configuration loading and updating"""
    print("Testing region selector configuration operations...")
    
    # Test loading current config
    print("\n1. Loading current configuration...")
    config = load_current_config()
    print(f"   Current region: {config['region_name']}")
    print(f"   Current bounds: {config['aoi']}")
    
    # Test updating config
    print("\n2. Testing config update...")
    test_region = "Test Andaman"
    test_bounds = {
        'min_lat': 11.5,
        'max_lat': 13.5,
        'min_lon': 92.0,
        'max_lon': 94.0
    }
    
    success = update_config(
        test_region,
        test_bounds['min_lat'],
        test_bounds['max_lat'],
        test_bounds['min_lon'],
        test_bounds['max_lon']
    )
    
    if success:
        print(f"   ✅ Configuration updated successfully for {test_region}")
        
        # Verify the update
        updated_config = load_current_config()
        print(f"   Verification - New region: {updated_config['region_name']}")
        print(f"   Verification - New bounds: {updated_config['aoi']}")
        
        # Restore original config
        original_success = update_config(
            config['region_name'],
            config['aoi']['min_lat'],
            config['aoi']['max_lat'],
            config['aoi']['min_lon'],
            config['aoi']['max_lon']
        )
        
        if original_success:
            print(f"   ✅ Original configuration restored")
        else:
            print(f"   ❌ Failed to restore original configuration")
    else:
        print(f"   ❌ Failed to update configuration")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_config_operations()