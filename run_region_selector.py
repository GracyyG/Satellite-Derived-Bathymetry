#!/usr/bin/env python3
"""
SDB Region Selector

Interactive region selector for Satellite-Derived Bathymetry pipeline.
Supports both Streamlit web interface and CLI mode.

Usage:
  python run_region_selector.py              # Streamlit web interface
  python run_region_selector.py --cli        # Command-line interface
  python -m streamlit run run_region_selector.py  # Explicit Streamlit

Features:
- Interactive region selection with map preview
- Dynamic configuration updates
- Pipeline integration
- Progress monitoring
- Region-specific output management
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'region_run_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_current_config() -> Dict:
    """Load the current region configuration"""
    config_path = project_root / "config" / "location_config.json"
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            # Return default configuration
            default_config = {
                "region_name": "lakshadweep",
                "aoi": {
                    "min_lat": 10.75,
                    "max_lat": 10.95,
                    "min_lon": 72.35,
                    "max_lon": 72.65
                }
            }
            return default_config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def update_config(region_name: str, min_lat: float, max_lat: float, 
                 min_lon: float, max_lon: float) -> bool:
    """Update the region configuration"""
    try:
        config_path = project_root / "config" / "location_config.json"
        config_path.parent.mkdir(exist_ok=True)
        
        # Create new configuration
        new_config = {
            "region_name": region_name.lower().replace(" ", "_"),
            "aoi": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            }
        }
        
        # Add region-specific entry
        region_slug = region_name.lower().replace(" ", "_")
        new_config[region_slug] = {
            "name": region_name,
            "bbox": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            }
        }
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        logger.info(f"Configuration updated for {region_name}")
        print(f"[OK] Configuration updated for {region_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        print(f"[ERROR] Failed to update configuration: {e}")
        return False

def run_pipeline() -> bool:
    """Run the SDB pipeline for the current region"""
    try:
        print("[INFO] Running SDB pipeline...")
        logger.info("Starting SDB pipeline execution...")
        
        # Import and run the main pipeline
        from main_pipeline import main as run_main_pipeline
        
        # Execute pipeline
        success = run_main_pipeline()
        
        if success:
            logger.info("Pipeline executed successfully")
            print("[OK] Pipeline completed successfully!")
        else:
            logger.error("Pipeline execution failed")
            print("[ERROR] Pipeline execution failed")
        
        return success
        
    except ImportError:
        # Fallback to running pipeline script
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "main_pipeline.py"
            ], cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Pipeline executed successfully")
                print("[OK] Pipeline completed successfully!")
                return True
            else:
                logger.error(f"Pipeline failed: {result.stderr}")
                print(f"[ERROR] Pipeline failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            print(f"[ERROR] Error running pipeline: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        print(f"[ERROR] Error running pipeline: {e}")
        return False

def get_processed_data_status(region_name: str) -> Dict:
    """Check if processed data exists for the region"""
    region_slug = region_name.lower().replace(" ", "_")
    
    # Check for processed data
    processed_data_path = Path("data/sentinel") / region_slug / "processed/training_data/features.npy"
    models_path = Path("models") / region_slug
    
    status = {
        "has_processed_data": processed_data_path.exists(),
        "has_models": models_path.exists() and any(models_path.glob("*.joblib")),
        "data_path": str(processed_data_path),
        "models_path": str(models_path)
    }
    
    return status

def cli_interface():
    """Command-line interface for region selection"""
    print("=" * 60)
    print("SDB REGION SELECTOR - CLI MODE")
    print("=" * 60)
    
    # Show current configuration
    config = load_current_config()
    if config:
        print(f"\\nCurrent Region: {config.get('region_name', 'Not set')}")
        aoi = config.get('aoi', {})
        print(f"Area of Interest:")
        print(f"  Latitude:  {aoi.get('min_lat', 'N/A')}° to {aoi.get('max_lat', 'N/A')}°")
        print(f"  Longitude: {aoi.get('min_lon', 'N/A')}° to {aoi.get('max_lon', 'N/A')}°")
    
    # Available regions with presets
    regions = {
        "1": {"name": "Lakshadweep", "bounds": (10.75, 10.95, 72.35, 72.65)},
        "2": {"name": "Palk Strait", "bounds": (9.0, 10.5, 78.5, 80.0)},
        "3": {"name": "Goa", "bounds": (15.0, 15.8, 73.5, 74.5)},
        "4": {"name": "Kachchh", "bounds": (22.5, 23.5, 68.5, 70.0)},
        "5": {"name": "Andaman", "bounds": (11.5, 13.5, 92.0, 94.0)},
        "6": {"name": "Custom", "bounds": None}
    }
    
    print("\\nAvailable Regions:")
    for key, region in regions.items():
        if region["bounds"]:
            lat_range = f"{region['bounds'][0]}° to {region['bounds'][1]}°"
            lon_range = f"{region['bounds'][2]}° to {region['bounds'][3]}°"
            print(f"  {key}. {region['name']} (Lat: {lat_range}, Lon: {lon_range})")
        else:
            print(f"  {key}. {region['name']} (Define custom coordinates)")
    
    # Get user selection
    while True:
        choice = input("\\nSelect region (1-6) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("Goodbye!")
            return
        
        if choice in regions:
            selected = regions[choice]
            region_name = selected["name"]
            
            if selected["bounds"]:
                min_lat, max_lat, min_lon, max_lon = selected["bounds"]
            else:
                # Custom region input
                try:
                    print("\\nEnter custom coordinates:")
                    min_lat = float(input("Minimum Latitude: "))
                    max_lat = float(input("Maximum Latitude: "))
                    min_lon = float(input("Minimum Longitude: "))
                    max_lon = float(input("Maximum Longitude: "))
                    region_name = input("Region Name: ").strip()
                except ValueError:
                    print("[ERROR] Invalid coordinates. Please try again.")
                    continue
            
            # Update configuration
            print(f"\\nUpdating configuration for {region_name}...")
            success = update_config(region_name, min_lat, max_lat, min_lon, max_lon)
            
            if success:
                # Check data status
                status = get_processed_data_status(region_name)
                if not status["has_processed_data"]:
                    logger.info("No processed data found for new region...")
                    print("[INFO] No processed data found for new region...")
                    print("[INFO] For new regions, you need to:")
                    print("[INFO] 1. Download Sentinel-2 data for the region")
                    print("[INFO] 2. Run data preprocessing")
                    print("[INFO] 3. Train models")
                    print("[INFO] Running visualization with synthetic data as demonstration...")
                
                # Ask if user wants to run pipeline
                run_choice = input("\\nRun pipeline for this region? (y/N): ").strip().lower()
                if run_choice == 'y':
                    run_pipeline()
                else:
                    print("[INFO] Configuration saved. You can run the pipeline later.")
                
                # Show results location
                region_slug = region_name.lower().replace(" ", "_")
                results_path = f"outputs/{region_slug}/"
                print(f"[INFO] Results saved to: {results_path}")
            
            break
        else:
            print("[ERROR] Invalid selection. Please try again.")

def streamlit_interface():
    """Streamlit web interface for region selection"""
    try:
        import streamlit as st
        import folium
        from streamlit_folium import st_folium
        
        st.set_page_config(
            page_title="SDB Region Selector",
            page_icon="🌊",
            layout="wide"
        )
        
        st.title("🌊 Satellite-Derived Bathymetry Region Selector")
        
        # Load current configuration
        config = load_current_config()
        
        # Sidebar for region selection
        st.sidebar.header("Region Configuration")
        
        # Preset regions
        preset_regions = {
            "Lakshadweep": (10.75, 10.95, 72.35, 72.65),
            "Palk Strait": (9.0, 10.5, 78.5, 80.0), 
            "Goa": (15.0, 15.8, 73.5, 74.5),
            "Kachchh": (22.5, 23.5, 68.5, 70.0),
            "Andaman": (11.5, 13.5, 92.0, 94.0),
            "Custom": None
        }
        
        selected_preset = st.sidebar.selectbox(
            "Choose a preset region:",
            list(preset_regions.keys()),
            index=0 if config.get('region_name') not in [k.lower().replace(' ', '_') for k in preset_regions.keys()] 
                  else [k.lower().replace(' ', '_') for k in preset_regions.keys()].index(config.get('region_name', 'lakshadweep'))
        )
        
        if preset_regions[selected_preset]:
            min_lat, max_lat, min_lon, max_lon = preset_regions[selected_preset]
            region_name = selected_preset
        else:
            st.sidebar.subheader("Custom Region")
            region_name = st.sidebar.text_input("Region Name", value=config.get('region_name', ''))
            aoi = config.get('aoi', {})
            min_lat = st.sidebar.number_input("Min Latitude", value=aoi.get('min_lat', 0.0))
            max_lat = st.sidebar.number_input("Max Latitude", value=aoi.get('max_lat', 0.0))
            min_lon = st.sidebar.number_input("Min Longitude", value=aoi.get('min_lon', 0.0))
            max_lon = st.sidebar.number_input("Max Longitude", value=aoi.get('max_lon', 0.0))
        
        # Update button
        if st.sidebar.button("Update Configuration"):
            success = update_config(region_name, min_lat, max_lat, min_lon, max_lon)
            if success:
                st.sidebar.success(f"Configuration updated for {region_name}")
                st.rerun()
            else:
                st.sidebar.error("Failed to update configuration")
        
        # Main area - Map
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Region Map")
            
            # Create map centered on region
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            
            # Add rectangle for AOI
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                popup=f"{region_name} AOI",
                color="red",
                weight=2,
                fill=True,
                fillOpacity=0.2
            ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
        
        with col2:
            st.subheader("Current Configuration")
            st.write(f"**Region:** {region_name}")
            st.write(f"**Latitude:** {min_lat}° to {max_lat}°")
            st.write(f"**Longitude:** {min_lon}° to {max_lon}°")
            
            # Data status
            st.subheader("Data Status")
            status = get_processed_data_status(region_name)
            
            if status["has_processed_data"]:
                st.success("✅ Processed data available")
            else:
                st.warning("⚠️ No processed data found")
            
            if status["has_models"]:
                st.success("✅ Trained models available")
            else:
                st.warning("⚠️ No trained models found")
            
            # Pipeline controls
            st.subheader("Pipeline Control")
            
            if st.button("🚀 Run Pipeline"):
                with st.spinner("Running pipeline..."):
                    success = run_pipeline()
                    if success:
                        st.success("Pipeline completed successfully!")
                    else:
                        st.error("Pipeline execution failed")
            
            # Results
            region_slug = region_name.lower().replace(" ", "_")
            results_path = Path(f"outputs/{region_slug}")
            if results_path.exists():
                st.subheader("Results")
                st.write(f"📁 Output directory: `{results_path}`")
                
                # List some key output files
                key_files = [
                    "final_showcase",
                    "visualizations", 
                    "executed_notebooks"
                ]
                
                for file_pattern in key_files:
                    matching = list(results_path.glob(f"**/*{file_pattern}*"))
                    if matching:
                        st.write(f"📄 {file_pattern}: {len(matching)} files")
    
    except ImportError:
        print("[ERROR] Streamlit not installed. Please install with: pip install streamlit")
        print("[INFO] Falling back to CLI mode...")
        cli_interface()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SDB Region Selector")
    parser.add_argument("--cli", action="store_true", help="Use command-line interface")
    args = parser.parse_args()
    
    # Ensure logs directory exists
    (project_root / "logs").mkdir(exist_ok=True)
    
    if args.cli:
        cli_interface()
    else:
        # Try to run Streamlit interface
        streamlit_interface()

if __name__ == "__main__":
    main()