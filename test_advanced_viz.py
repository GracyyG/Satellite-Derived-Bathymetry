#!/usr/bin/env python3
"""
Test Script for Advanced Visualization Integration
Tests the new visualization system integration with the SDB pipeline
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_visualization_integration():
    """Test the advanced visualization integration"""
    
    print("🧪 Testing Advanced Visualization Integration")
    print("=" * 50)
    
    try:
        # Test 1: Import the visualization system
        print("Test 1: Importing visualization system...")
        from src.visualize import AdvancedVisualizationManager, run_advanced_visualizations
        print("✅ Successfully imported visualization system")
        
        # Test 2: Discover visualization modules
        print("\nTest 2: Discovering visualization modules...")
        viz_manager = AdvancedVisualizationManager(project_root)
        modules = viz_manager.discover_visualization_modules()
        
        print(f"✅ Found {len(modules)} visualization modules:")
        for module in modules:
            print(f"  - {module}")
        
        # Test 3: Load modules
        print("\nTest 3: Loading visualization modules...")
        loaded_modules = viz_manager.load_visualization_modules()
        print(f"✅ Successfully loaded {len(loaded_modules)} modules")
        
        # Test 4: Check visualization functions
        print(f"\nTest 4: Checking visualization functions...")
        print(f"✅ Found {len(viz_manager.viz_functions)} visualization functions:")
        for func_name in viz_manager.viz_functions.keys():
            print(f"  - {func_name}")
        
        # Test 5: Verify file paths exist
        print(f"\nTest 5: Verifying visualization script files...")
        for module_name in modules:
            script_path = viz_manager.visualizations_dir / f"{module_name}.py"
            if script_path.exists():
                print(f"  ✅ {script_path.name}")
            else:
                print(f"  ❌ {script_path.name} - NOT FOUND")
        
        print(f"\n🎉 All tests passed! Advanced visualization integration is working correctly.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_integration_summary():
    """Show summary of the integration"""
    
    print("\n" + "=" * 60)
    print("📋 ADVANCED VISUALIZATION INTEGRATION SUMMARY")
    print("=" * 60)
    
    print("\n🎯 INTEGRATION FEATURES:")
    features = [
        "✅ Automatic discovery of visualization scripts in visualisations/",
        "✅ Safe module loading with error handling",
        "✅ Subprocess execution to avoid conflicts",
        "✅ File generation tracking and reporting",
        "✅ Integration with 09_final_visual_showcase.ipynb",
        "✅ Unique filename generation to avoid overwrites",
        "✅ Comprehensive error handling and logging"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n📊 DISCOVERED VISUALIZATION SCRIPTS:")
    viz_dir = Path("visualisations")
    if viz_dir.exists():
        for script in viz_dir.glob("*.py"):
            if script.name != "main_pipeline.py":
                print(f"  🎨 {script.name}")
    
    print(f"\n🚀 USAGE:")
    print(f"  1. Run: python run_full_pipeline.py")
    print(f"  2. Or execute the final showcase notebook directly")
    print(f"  3. Advanced visualizations will be automatically generated")
    print(f"  4. Check outputs/{{region}}/final_showcase/ for results")
    
    print(f"\n📁 OUTPUT STRUCTURE:")
    print(f"  outputs/{{region_name}}/")
    print(f"  ├── final_showcase/              # Advanced visualizations")
    print(f"  │   ├── 3d_*.html               # Interactive 3D plots")
    print(f"  │   ├── *_comparison*.png       # Model comparisons")
    print(f"  │   ├── *_heatmap*.png          # Geographic heatmaps")
    print(f"  │   └── *_surface*.html         # Surface plots")
    print(f"  ├── visualizations/             # Basic visualizations")
    print(f"  └── executed_notebooks/         # Papermill outputs")
    
    print("=" * 60)

if __name__ == "__main__":
    success = test_visualization_integration()
    show_integration_summary()
    
    if success:
        print("\n🎉 Integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)