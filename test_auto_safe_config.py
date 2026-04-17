#!/usr/bin/env python3
"""
Test Script for SAFE File Auto-Configuration
Demonstrates the complete integration between coordinate updates and SAFE file selection
"""

import json
import sys
import time
from pathlib import Path

def test_auto_configuration():
    """Test the complete auto-configuration system"""
    
    print("🧪 TESTING SAFE FILE AUTO-CONFIGURATION SYSTEM")
    print("=" * 60)
    
    # Test regions with coordinates
    test_regions = [
        {
            "name": "Goa_Test", 
            "coords": {"min_lat": 15.0, "max_lat": 15.8, "min_lon": 73.7, "max_lon": 74.3},
            "expected_safe": "S2A_MSIL2A_20251106T053241_N0511_R105_T43PCS_20251106T083512.SAFE.zip"
        },
        {
            "name": "Kachchh_Test", 
            "coords": {"min_lat": 22.5, "max_lat": 24.0, "min_lon": 68.0, "max_lon": 71.0},
            "expected_safe": "S2C_MSIL1C_20251110T055101_N0511_R048_T42QWK_20251110T074050.SAFE.zip"
        },
        {
            "name": "Rameswaram_PalkStrait", 
            "coords": {"min_lat": 8.5, "max_lat": 10.0, "min_lon": 78.5, "max_lon": 80.5},
            "expected_safe": "S2B_MSIL2A_20251031T045839_N0511_R119_T44PKR_20251031T072153.SAFE.zip"
        }
    ]
    
    results = []
    
    for i, region in enumerate(test_regions, 1):
        print(f"\n🧪 TEST {i}/3: {region['name']}")
        print("-" * 40)
        
        # Update location config
        config = {
            "region_name": region["name"],
            "aoi": region["coords"]
        }
        
        with open("config/location_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Updated config for {region['name']}")
        
        # Run auto-configuration
        import subprocess
        try:
            result = subprocess.run([
                sys.executable, "auto_safe_config.py"
            ], capture_output=True, text=True, cwd=".")
            
            success = result.returncode == 0
            
            if success:
                # Check if correct SAFE file was selected
                if region["expected_safe"] in result.stdout:
                    print(f"✅ Correct SAFE file selected: {region['expected_safe']}")
                    test_result = "PASS"
                else:
                    print(f"❌ Wrong SAFE file selected")
                    test_result = "FAIL"
            else:
                print(f"❌ Configuration failed: {result.stderr}")
                test_result = "ERROR"
                
        except Exception as e:
            print(f"❌ Exception during configuration: {e}")
            test_result = "ERROR"
        
        results.append({
            "region": region["name"],
            "result": test_result,
            "expected": region["expected_safe"]
        })
        
        # Small delay between tests
        time.sleep(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["result"] == "PASS")
    total = len(results)
    
    for result in results:
        status_icon = "✅" if result["result"] == "PASS" else "❌"
        print(f"{status_icon} {result['region']}: {result['result']}")
    
    print(f"\n🏆 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Auto-configuration system is working perfectly!")
    else:
        print("⚠️ Some tests failed. Please check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = test_auto_configuration()
    sys.exit(0 if success else 1)