#!/usr/bin/env python3
"""
Map ICESat-2 bathymetry points to Sentinel-2 processed grid
Creates training data by spatially matching satellite bathymetry with Sentinel-2 features
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

class ICESat2Mapper:
    """Maps ICESat-2 bathymetry points to Sentinel-2 features grid"""
    
    def __init__(self, region_name):
        self.region_name = region_name
        self.project_root = Path(__file__).parent.parent
        
        # Paths
        self.icesat_data_path = self.project_root / f"data/icesat2/{region_name}"
        self.sentinel_data_path = self.project_root / f"data/sentinel/{region_name}/processed"
        self.output_path = self.project_root / f"data/processed/{region_name}/training_data"
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[OK] ICESat-2 Mapper initialized for region: {region_name}")
    
    def load_icesat2_points(self):
        """Load ICESat-2 bathymetry points"""
        
        points_file = self.icesat_data_path / "bathymetry_points.csv"
        
        if not points_file.exists():
            raise FileNotFoundError(f"ICESat-2 points not found: {points_file}")
        
        points_df = pd.read_csv(points_file)
        
        # Filter for valid bathymetry (negative elevations)
        bathy_points = points_df[points_df['elevation'] < 0].copy()
        bathy_points['depth'] = -bathy_points['elevation']  # Convert to positive depth
        
        print(f"[OK] Loaded {len(bathy_points)} ICESat-2 bathymetry points")
        print(f"  Depth range: {bathy_points['depth'].min():.1f}m to {bathy_points['depth'].max():.1f}m")
        
        return bathy_points
    
    def load_sentinel2_features(self):
        """Load processed Sentinel-2 features and coordinates"""
        
        # Look for feature files
        feature_files = list(self.sentinel_data_path.glob("*features*.npy"))
        coord_files = list(self.sentinel_data_path.glob("*coordinates*.npy"))
        
        if not feature_files:
            raise FileNotFoundError(f"No Sentinel-2 features found in {self.sentinel_data_path}")
        
        # Load features
        features_file = feature_files[0]
        features = np.load(features_file)
        
        print(f"[OK] Loaded Sentinel-2 features: {features.shape}")
        
        # Load or generate coordinates
        if coord_files:
            coords_file = coord_files[0]
            coords = np.load(coords_file)
            print(f"[OK] Loaded coordinates: {coords.shape}")
        else:
            # Generate coordinates from raster if available
            coords = self._generate_coordinates_from_raster()
        
        return features, coords
    
    def _generate_coordinates_from_raster(self):
        """Generate coordinates from Sentinel-2 raster files"""
        
        # Look for any processed raster file to get geo-reference
        raster_files = list(self.sentinel_data_path.glob("*.tif"))
        
        if not raster_files:
            raise FileNotFoundError("No raster files found to generate coordinates")
        
        # Use first raster file for geo-reference
        with rasterio.open(raster_files[0]) as src:
            height, width = src.height, src.width
            transform = src.transform
            
            # Generate coordinate grid
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            
            # Convert pixel coordinates to geographic coordinates
            lons, lats = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
            
            coords = np.column_stack((lats, lons))
            
            print(f"[OK] Generated coordinates from raster: {coords.shape}")
            
            return coords
    
    def spatial_matching(self, icesat_points, sentinel_coords, max_distance=30):
        """Match ICESat-2 points to nearest Sentinel-2 pixels"""
        
        print(f"[INFO] Performing spatial matching (max distance: {max_distance}m)...")
        
        # Create coordinate arrays
        icesat_coords = icesat_points[['lat', 'lon']].values
        
        # Build KD-Tree for fast nearest neighbor search
        tree = cKDTree(sentinel_coords)
        
        # Find nearest Sentinel-2 pixels for each ICESat-2 point
        # Convert max_distance from meters to degrees (rough approximation)
        max_dist_deg = max_distance / 111000  # ~111km per degree
        
        distances, indices = tree.query(icesat_coords, distance_upper_bound=max_dist_deg)
        
        # Filter valid matches
        valid_mask = distances < max_dist_deg
        valid_icesat_idx = np.where(valid_mask)[0]
        valid_sentinel_idx = indices[valid_mask]
        
        print(f"[OK] Found {len(valid_icesat_idx)} valid spatial matches")
        print(f"  Match rate: {len(valid_icesat_idx)/len(icesat_points)*100:.1f}%")
        
        return valid_icesat_idx, valid_sentinel_idx, distances[valid_mask]
    
    def create_training_data(self, features, icesat_points, icesat_idx, sentinel_idx):
        """Create matched training dataset"""
        
        # Extract matched features and depths
        matched_features = features[sentinel_idx]
        matched_depths = icesat_points.iloc[icesat_idx]['depth'].values
        
        # Get coordinates for matched points
        matched_coords = icesat_points.iloc[icesat_idx][['lat', 'lon']].values
        
        print(f"[OK] Created training dataset:")
        print(f"  Features shape: {matched_features.shape}")
        print(f"  Depths shape: {matched_depths.shape}")
        print(f"  Depth range: {matched_depths.min():.1f}m to {matched_depths.max():.1f}m")
        print(f"  Mean depth: {matched_depths.mean():.1f}m")
        
        return matched_features, matched_depths, matched_coords
    
    def save_training_data(self, features, depths, coords):
        """Save training data for model training"""
        
        # Save numpy arrays
        features_file = self.output_path / "features.npy"
        depths_file = self.output_path / "depths.npy"
        coords_file = self.output_path / "coordinates.npy"
        
        np.save(features_file, features)
        np.save(depths_file, depths)
        np.save(coords_file, coords)
        
        print(f"[OK] Saved training data:")
        print(f"  Features: {features_file}")
        print(f"  Depths: {depths_file}")
        print(f"  Coordinates: {coords_file}")
        
        # Create metadata
        metadata = {
            'region': self.region_name,
            'n_samples': len(features),
            'n_features': features.shape[1],
            'depth_range': [float(depths.min()), float(depths.max())],
            'depth_mean': float(depths.mean()),
            'depth_std': float(depths.std()),
            'created_at': pd.Timestamp.now().isoformat(),
            'data_source': 'ICESat-2 ATL03 + Sentinel-2'
        }
        
        metadata_file = self.output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata: {metadata_file}")
        
        return metadata
    
    def create_validation_split(self, features, depths, coords, test_size=0.2):
        """Create train/validation split"""
        
        n_samples = len(features)
        n_test = int(n_samples * test_size)
        
        # Random split
        indices = np.random.permutation(n_samples)
        train_idx = indices[:-n_test] if n_test > 0 else indices
        test_idx = indices[-n_test:] if n_test > 0 else []
        
        # Save splits
        split_data = {
            'train': {
                'features': features[train_idx],
                'depths': depths[train_idx],
                'coords': coords[train_idx]
            }
        }
        
        if len(test_idx) > 0:
            split_data['test'] = {
                'features': features[test_idx],
                'depths': depths[test_idx],
                'coords': coords[test_idx]
            }
        
        # Save train data
        train_dir = self.output_path / "train"
        train_dir.mkdir(exist_ok=True)
        
        np.save(train_dir / "features.npy", split_data['train']['features'])
        np.save(train_dir / "depths.npy", split_data['train']['depths'])
        np.save(train_dir / "coordinates.npy", split_data['train']['coords'])
        
        # Save test data if exists
        if 'test' in split_data:
            test_dir = self.output_path / "test"
            test_dir.mkdir(exist_ok=True)
            
            np.save(test_dir / "features.npy", split_data['test']['features'])
            np.save(test_dir / "depths.npy", split_data['test']['depths'])
            np.save(test_dir / "coordinates.npy", split_data['test']['coords'])
            
            print(f"[OK] Created train/test split:")
            print(f"  Train samples: {len(split_data['train']['features'])}")
            print(f"  Test samples: {len(split_data['test']['features'])}")
        else:
            print(f"[OK] Created training data: {len(split_data['train']['features'])} samples")

def main():
    parser = argparse.ArgumentParser(description="Map ICESat-2 data to Sentinel-2 grid")
    parser.add_argument("--region", required=True, help="Region name (e.g., kachchh)")
    parser.add_argument("--max-distance", type=float, default=30, 
                       help="Maximum matching distance in meters (default: 30)")
    
    args = parser.parse_args()
    
    try:
        print("="*80)
        print(f"MAPPING ICESAT-2 DATA TO SENTINEL-2 GRID: {args.region.upper()}")
        print("="*80)
        
        # Initialize mapper
        mapper = ICESat2Mapper(args.region)
        
        # Load data
        print("\n1. Loading ICESat-2 bathymetry points...")
        icesat_points = mapper.load_icesat2_points()
        
        print("\n2. Loading Sentinel-2 features...")
        features, coords = mapper.load_sentinel2_features()
        
        print("\n3. Performing spatial matching...")
        icesat_idx, sentinel_idx, distances = mapper.spatial_matching(
            icesat_points, coords, args.max_distance
        )
        
        if len(icesat_idx) == 0:
            print("[ERROR] No spatial matches found! Try increasing --max-distance")
            return
        
        print("\n4. Creating training dataset...")
        train_features, train_depths, train_coords = mapper.create_training_data(
            features, icesat_points, icesat_idx, sentinel_idx
        )
        
        print("\n5. Saving training data...")
        metadata = mapper.save_training_data(train_features, train_depths, train_coords)
        
        print("\n6. Creating train/validation split...")
        mapper.create_validation_split(train_features, train_depths, train_coords)
        
        print(f"\n{'='*80}")
        print("MAPPING COMPLETE!")
        print(f"{'='*80}")
        print(f"Region: {args.region}")
        print(f"Training samples: {metadata['n_samples']}")
        print(f"Features: {metadata['n_features']}")
        print(f"Depth range: {metadata['depth_range'][0]:.1f}m - {metadata['depth_range'][1]:.1f}m")
        print(f"Mean depth: {metadata['depth_mean']:.1f}m")
        
        print(f"\n[NEXT STEP] Retrain models with real data:")
        print(f"cd notebooks && python -m jupyter notebook 03_model_training.ipynb")
        
    except Exception as e:
        print(f"[ERROR] Mapping failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()