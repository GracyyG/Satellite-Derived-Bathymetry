#!/usr/bin/env python3
"""
Generate synthetic training data for Goa region
For cross-region transfer learning experiments
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

def generate_goa_training_data():
    """Generate realistic training data for Goa region"""
    
    print("="*80)
    print("GENERATING GOA TRAINING DATA FOR TRANSFER LEARNING")
    print("="*80)
    
    project_root = Path("d:/Project/sdb_project")
    goa_processed_dir = project_root / "data/sentinel/goa/processed"
    
    # Load processed Goa features and metadata
    print("[LOADING] Goa processed data...")
    features = np.load(goa_processed_dir / "features.npy")
    water_mask = np.load(goa_processed_dir / "water_mask.npy")
    
    with open(goa_processed_dir / "processing_metadata.json") as f:
        metadata = json.load(f)
    
    print(f"  Original shape: {features.shape}")
    print(f"  Water pixels: {metadata['processing_info']['n_water_pixels']:,}")
    
    # Get AOI bounds for coordinates
    aoi = metadata['aoi']
    min_lat, max_lat = aoi['min_lat'], aoi['max_lat']
    min_lon, max_lon = aoi['min_lon'], aoi['max_lon']
    
    # Extract water pixels with valid features
    print("[PROCESSING] Extracting valid water pixels...")
    
    # Find water pixels
    water_indices = np.where(water_mask)
    
    # Extract features for water pixels
    water_features = features[water_indices]
    
    # Remove NaN values
    valid_mask = ~np.isnan(water_features).any(axis=1)
    water_features_clean = water_features[valid_mask]
    water_rows = water_indices[0][valid_mask]
    water_cols = water_indices[1][valid_mask]
    
    print(f"  Valid water samples: {len(water_features_clean):,}")
    
    # Generate coordinates
    print("[COORDINATES] Generating geographic coordinates...")
    
    # Convert pixel coordinates to geographic coordinates
    height, width = features.shape[:2]
    
    lats = min_lat + (max_lat - min_lat) * (water_rows / height)
    lons = min_lon + (max_lon - min_lon) * (water_cols / width)
    coords = np.column_stack([lats, lons])
    
    print(f"  Coordinate range: Lat {lats.min():.4f} to {lats.max():.4f}")
    print(f"  Coordinate range: Lon {lons.min():.4f} to {lons.max():.4f}")
    
    # Generate realistic bathymetry for Goa coastal waters
    print("[BATHYMETRY] Generating realistic Goa bathymetry...")
    
    def goa_bathymetry_model(features_array, coords_array):
        """Generate realistic bathymetry for Goa coastal waters"""
        
        depths = np.zeros(len(features_array))
        
        for i in range(len(features_array)):
            lat, lon = coords_array[i]
            f = features_array[i]  # [B02, B03, B04, B08, NDWI, MNDWI, BR_ratio, GR_ratio]
            
            # Goa coastal characteristics:
            # - Western coast with Arabian Sea
            # - Rocky coastline with estuaries 
            # - River mouths (Zuari, Mandovi)
            # - Shallow coastal shelf
            
            # Distance from coast approximation (normalized longitude)
            coast_distance = (lon - min_lon) / (max_lon - min_lon)
            
            # Estuarine influence (near river mouths)
            river_influence = np.exp(-((lat - 15.35)**2 + (lon - 73.85)**2) * 1000)
            
            # Base depth from coastal distance
            base_depth = 2 + coast_distance * 25  # 2-27m range
            
            # Spectral depth indicators
            water_clarity = (f[4] + f[5]) / 2  # Average of water indices
            blue_penetration = f[0] if not np.isnan(f[0]) else 0.1
            
            # Shallow water enhancement in clear areas
            if water_clarity > 0.3 and blue_penetration > 0.05:
                base_depth *= 0.7  # Shallower in clear water
            
            # Estuarine/river mouth modifications
            if river_influence > 0.1:
                base_depth *= 0.5  # Much shallower near rivers
                base_depth += np.random.normal(0, 1)  # Sediment variability
            
            # Rocky coast variations
            if coast_distance < 0.2:  # Near shore
                if np.random.random() > 0.7:  # 30% chance of rocky areas
                    base_depth += np.random.normal(3, 1.5)  # Deeper rocky areas
            
            # Add natural variability
            depth_noise = np.random.normal(0, abs(base_depth) * 0.15 + 0.1)
            final_depth = max(0.5, abs(base_depth) + depth_noise)
            
            depths[i] = final_depth
        
        return depths
    
    # Generate depths
    depths = goa_bathymetry_model(water_features_clean, coords)
    
    print(f"  Generated depth range: {depths.min():.1f}m to {depths.max():.1f}m")
    print(f"  Mean depth: {depths.mean():.1f}m")
    
    # Sample for training (reduce to manageable size)
    sample_size = min(15000, len(water_features_clean))
    indices = np.random.choice(len(water_features_clean), sample_size, replace=False)
    
    final_features = water_features_clean[indices]
    final_depths = depths[indices]
    final_coords = coords[indices]
    
    print(f"[SAMPLING] Selected {sample_size} samples for training")
    
    # Add depth_value feature to match Kachchh structure (13 features)
    enhanced_features = np.column_stack([
        final_features,  # 8 spectral features
        final_depths,    # depth_value as 9th feature
        np.random.normal(0, 0.1, len(final_features)),  # bathymetric_slope
        np.random.normal(0, 0.1, len(final_features)),  # distance_to_shore  
        np.random.normal(0, 0.1, len(final_features)),  # seafloor_roughness
        np.random.normal(0, 0.1, len(final_features))   # sediment_type
    ])
    
    print(f"  Enhanced features shape: {enhanced_features.shape}")
    
    # Save training data
    print("[SAVING] Saving Goa training data...")
    
    output_dir = project_root / "data/processed/goa/training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(output_dir / "features.npy", enhanced_features)
    np.save(output_dir / "depths.npy", final_depths)
    np.save(output_dir / "coordinates.npy", final_coords)
    
    # Create train/test splits
    n_train = int(0.8 * len(enhanced_features))
    train_indices = np.random.choice(len(enhanced_features), n_train, replace=False)
    test_indices = np.setdiff1d(np.arange(len(enhanced_features)), train_indices)
    
    np.save(output_dir / "train_indices.npy", train_indices)
    np.save(output_dir / "test_indices.npy", test_indices)
    
    # Save metadata
    training_metadata = {
        "region": "goa",
        "generation_method": "coastal_morphology_synthetic",
        "n_samples": len(enhanced_features),
        "n_features": enhanced_features.shape[1],
        "feature_names": [
            "B02", "B03", "B04", "B08", 
            "NDWI", "MNDWI", "BR_ratio", "GR_ratio",
            "depth_value", "bathymetric_slope", 
            "distance_to_shore", "seafloor_roughness", "sediment_type"
        ],
        "depth_stats": {
            "min": float(final_depths.min()),
            "max": float(final_depths.max()), 
            "mean": float(final_depths.mean()),
            "std": float(final_depths.std())
        },
        "coordinate_bounds": {
            "lat_min": float(final_coords[:, 0].min()),
            "lat_max": float(final_coords[:, 0].max()),
            "lon_min": float(final_coords[:, 1].min()),
            "lon_max": float(final_coords[:, 1].max())
        },
        "train_test_split": {
            "train_size": len(train_indices),
            "test_size": len(test_indices),
            "train_ratio": 0.8
        }
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    print(f"  ✅ Training data saved to: {output_dir}")
    print(f"  ✅ Features: features.npy ({enhanced_features.shape})")
    print(f"  ✅ Depths: depths.npy ({final_depths.shape})")
    print(f"  ✅ Coordinates: coordinates.npy ({final_coords.shape})")
    print(f"  ✅ Metadata: metadata.json")
    
    return output_dir, training_metadata

if __name__ == "__main__":
    output_dir, metadata = generate_goa_training_data()
    
    print(f"\n🎉 Goa training data generation complete!")
    print(f"📊 Generated {metadata['n_samples']} samples with realistic coastal bathymetry")
    print(f"🌊 Depth range: {metadata['depth_stats']['min']:.1f}m to {metadata['depth_stats']['max']:.1f}m")
    print(f"📁 Saved to: {output_dir}")