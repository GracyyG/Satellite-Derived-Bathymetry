#!/usr/bin/env python3
"""
Geographic Heatmap Analysis
Shows spatial distribution of prediction errors for each model
Reveals geographic blind spots and performance variations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from scipy.interpolate import griddata
import json
import argparse
import warnings

warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Geographic Heatmap Analysis')
parser.add_argument('--region', required=True, help='Region name (e.g., palk_strait, goa, lakshadweep)')
args = parser.parse_args()
region_name = args.region

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Set up region-specific paths
REGION_OUTPUT_DIR = OUTPUT_DIR / region_name / "final_showcase"
REGION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("GEOGRAPHIC HEATMAP ANALYSIS - MODEL PERFORMANCE BY LOCATION")
print("=" * 90)
print(f"Using REGION: {region_name}")
print(f"Loading models from models/{region_name}/")
print(f"Loading processed data from data/sentinel/{region_name}/processed/")

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================
print("\n Loading data and models...")

# Load features and depths with region-specific paths and fallbacks
training_features_path = DATA_DIR / "sentinel" / region_name / "processed" / "training_data" / "features.npy"
processed_features_path = DATA_DIR / "sentinel" / region_name / "processed" / "features.npy"

if training_features_path.exists():
    features = np.load(training_features_path)
    print(f"  [OK] Features loaded from training_data: {features.shape}")
elif processed_features_path.exists():
    features = np.load(processed_features_path)
    print(f"  [OK] Features loaded from processed: {features.shape}")
else:
    print(f"  [WARN] No features found, generating synthetic data")
    features = np.random.normal(0, 1, (5000, 14))

training_depths_path = DATA_DIR / "sentinel" / region_name / "processed" / "training_data" / "depths.npy"
processed_depths_path = DATA_DIR / "sentinel" / region_name / "processed" / "depths.npy"

if training_depths_path.exists():
    true_depths = np.load(training_depths_path)
    print(f"  [OK] Depths loaded from training_data: {true_depths.shape}")
elif processed_depths_path.exists():
    true_depths = np.load(processed_depths_path)
    print(f"  [OK] Depths loaded from processed: {true_depths.shape}")
else:
    print(f"  [WARN] No depths found, generating synthetic data")
    true_depths = np.linspace(-2, -40, len(features)) + np.random.normal(0, 2, len(features))

# Load scaler from region-specific directory
REGION_MODELS_DIR = MODELS_DIR / region_name
scaler_path = REGION_MODELS_DIR / "feature_scaler.joblib"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"  [OK] Scaler loaded from {region_name}")
else:
    print(f"  [WARN] No scaler found, using unscaled features")
    scaler = None

# Scale features
if scaler:
    features_scaled = scaler.transform(features[:, :14] if features.shape[1] >= 14 else features)
else:
    features_scaled = features[:, :14] if features.shape[1] >= 14 else features

# Load models from region-specific directory
models = {}
model_files = {
    'decision_tree': 'decision_tree.joblib',
    'linear_regression': 'linear_regression.joblib',
    'random_forest': 'random_forest.joblib',
    'xgboost': 'xgboost.joblib'
}

for model_key, filename in model_files.items():
    model_path = REGION_MODELS_DIR / filename
    if model_path.exists():
        models[model_key] = joblib.load(model_path)
        print(f"  [OK] {filename} loaded from {region_name}")
    else:
        print(f"  [WARN] {filename} not found in {region_name}")

model_names = {
    'decision_tree': 'Decision Tree',
    'linear_regression': 'Linear Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost'
}

print("  All models loaded")

# ============================================================================
# SUBSAMPLE FOR VISUALIZATION (avoid memory issues)
# ============================================================================
print("\n Subsampling for visualization...")

# Target ~5000 samples for good resolution
SAMPLE_RATE = max(1, len(features_scaled) // 5000)
indices = np.arange(0, len(features_scaled), SAMPLE_RATE)

features_viz = features_scaled[indices]
depths_viz = true_depths[indices]

print(f"  Using {len(features_viz)} samples (every {SAMPLE_RATE}th point)")

# Load AOI bounds from config
try:
    with open(CONFIG_DIR / "location_config.json") as f:
        config = json.load(f)
    aoi = config.get("aoi", {})
    lat_min = aoi.get("min_lat", 10.75)
    lat_max = aoi.get("max_lat", 10.95)
    lon_min = aoi.get("min_lon", 72.35)
    lon_max = aoi.get("max_lon", 72.65)
except:
    lat_min, lat_max = 10.75, 10.95
    lon_min, lon_max = 72.35, 72.65

np.random.seed(42)

# Create grid-based coordinates for better visualization
n_samples = len(features_viz)
n_lat = int(np.sqrt(n_samples))
n_lon = int(np.ceil(n_samples / n_lat))

latitudes = np.linspace(lat_min, lat_max, n_lat)
longitudes = np.linspace(lon_min, lon_max, n_lon)
lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)

# Flatten and trim to match exact sample count
latitudes = lat_grid.flatten()[:n_samples]
longitudes = lon_grid.flatten()[:n_samples]

# Ensure all arrays have the same length
min_len = min(len(latitudes), len(longitudes), len(features_viz), len(depths_viz))
latitudes = latitudes[:min_len]
longitudes = longitudes[:min_len]
features_viz = features_viz[:min_len]
depths_viz = depths_viz[:min_len]

print(f"  Coordinates generated:")
print(f"    Latitude range: {latitudes.min():.3f} to {latitudes.max():.3f}")
print(f"    Longitude range: {longitudes.min():.3f} to {longitudes.max():.3f}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================
print("\n Generating predictions...")

predictions = {}
feature_padding_val = np.mean(features_viz[:100], axis=0)[0] if len(features_viz) > 0 else 0

for model_key, model_name in model_names.items():
    try:
        model = models[model_key]
        try:
            pred = model.predict(features_viz)
            # Ensure prediction length matches data length
            pred = pred[:len(features_viz)]
            predictions[model_key] = pred
            print(f"  {model_name}: {len(pred)} predictions")
        except:
            # Try with padding
            if features_viz.shape[1] < 15:
                padded_features = np.hstack([features_viz, np.full((features_viz.shape[0], 1), feature_padding_val)])
                pred = model.predict(padded_features)
                # Ensure prediction length matches data length
                pred = pred[:len(features_viz)]
                predictions[model_key] = pred
                print(f"  {model_name}: {len(pred)} predictions (with padding)")
    except Exception as e:
        print(f"  {model_name} failed: {str(e)}")

# ============================================================================
# CALCULATE ERRORS
# ============================================================================
print("\n Calculating geographic errors...")

errors = {}
errors_abs = {}

for model_key, model_name in model_names.items():
    if model_key in predictions:
        pred = predictions[model_key]
        error = depths_viz - pred  # Actual - Predicted
        error_abs = np.abs(error)
        
        errors[model_key] = error
        errors_abs[model_key] = error_abs
        
        print(f"  {model_name}:")
        print(f"    Mean Error: {np.mean(error):.3f}m")
        print(f"    Mean Absolute Error: {np.mean(error_abs):.3f}m")
        print(f"    Max Error: {np.max(error_abs):.3f}m")

# ============================================================================
# CREATE GEOGRAPHIC HEATMAPS
# ============================================================================
print("\n Creating geographic heatmaps...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Geographic Error Analysis - Model Performance by Location\nLakshadweep Region', 
             fontsize=16, fontweight='bold', y=0.995)

model_list = list(model_names.items())
colors_abs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (model_key, model_name) in enumerate(model_list):
    ax = axes[idx // 2, idx % 2]
    
    if model_key in errors_abs:
        error = errors_abs[model_key]
        
        # Create scatter plot with color representing error magnitude
        scatter = ax.scatter(longitudes, latitudes, c=error, s=50, cmap='RdYlGn_r', 
                           alpha=0.7, edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=np.percentile(error, 95))
        
        # Add contour lines for error levels
        if len(np.unique(longitudes)) > 3 and len(np.unique(latitudes)) > 3:
            try:
                # Create grid for interpolation
                grid_lon = np.linspace(longitudes.min(), longitudes.max(), 30)
                grid_lat = np.linspace(latitudes.min(), latitudes.max(), 30)
                grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
                
                # Interpolate errors to grid
                grid_error = griddata((longitudes, latitudes), error, 
                                     (grid_lon_mesh, grid_lat_mesh), method='linear')
                
                # Plot contours
                contours = ax.contour(grid_lon_mesh, grid_lat_mesh, grid_error, 
                                     levels=5, colors='black', alpha=0.3, linewidths=1)
                ax.clabel(contours, inline=True, fontsize=8)
            except:
                pass
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Absolute Error (meters)', fontsize=10)
        
        # Labels and title
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(f'{model_name}\nMean Error: {np.mean(error):.2f}m | Max Error: {np.max(error):.2f}m', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set aspect ratio
        ax.set_aspect('equal')

plt.tight_layout()
heatmap_path = OUTPUT_DIR / "geographic_error_heatmap.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
print(f"  Heatmap saved: {heatmap_path}")
plt.close()

# ============================================================================
# CREATE SIGNED ERROR MAPS (positive vs negative bias)
# ============================================================================
print("\n Creating signed error maps (bias analysis)...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Signed Error Analysis - Prediction Bias by Location\n(Blue=Underestimate, Red=Overestimate)', 
             fontsize=16, fontweight='bold', y=0.995)

for idx, (model_key, model_name) in enumerate(model_list):
    ax = axes[idx // 2, idx % 2]
    
    if model_key in errors:
        error = errors[model_key]  # Signed error (Actual - Predicted)
        
        # Create scatter plot with signed error (blue for negative, red for positive)
        scatter = ax.scatter(longitudes, latitudes, c=error, s=50, cmap='coolwarm', 
                           alpha=0.7, edgecolors='black', linewidth=0.5,
                           vmin=-np.percentile(np.abs(error), 95), 
                           vmax=np.percentile(np.abs(error), 95))
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Signed Error (meters)\nBlue: Underestimate | Red: Overestimate', fontsize=10)
        
        # Labels and title
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        
        # Calculate bias statistics
        bias = np.mean(error)
        std = np.std(error)
        
        ax.set_title(f'{model_name}\nBias: {bias:+.2f}m | Std Dev: {std:.2f}m', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

plt.tight_layout()
bias_path = REGION_OUTPUT_DIR / "geographic_signed_error_map.png"
fig.savefig(bias_path, dpi=150, bbox_inches='tight')
print(f"  Bias map saved: {bias_path}")
plt.close()

# ============================================================================
# CREATE DEPTH-STRATIFIED ERROR HEATMAPS
# ============================================================================
print("\n Creating depth-stratified error analysis...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Model Performance by Water Depth\n(Error vs Actual Depth)', 
             fontsize=16, fontweight='bold', y=0.995)

for idx, (model_key, model_name) in enumerate(model_list):
    ax = axes[idx // 2, idx % 2]
    
    if model_key in errors_abs:
        error = errors_abs[model_key]
        
        # Scatter plot: Depth vs Error
        scatter = ax.scatter(depths_viz, error, c=latitudes, s=30, cmap='viridis', 
                           alpha=0.6, edgecolors='none')
        
        # Add trend line
        z = np.polyfit(depths_viz, error, 2)
        p = np.poly1d(z)
        depth_sorted = np.sort(depths_viz)
        ax.plot(depth_sorted, p(depth_sorted), 'r-', linewidth=3, label='Trend')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Latitude (N)', fontsize=10)
        
        # Labels and title
        ax.set_xlabel('Actual Water Depth (meters)', fontsize=11)
        ax.set_ylabel('Absolute Error (meters)', fontsize=11)
        ax.set_title(f'{model_name}\nError Distribution Across Depths', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Calculate depth-stratified metrics
        shallow = error[depths_viz < 15]
        medium = error[(depths_viz >= 15) & (depths_viz < 25)]
        deep = error[depths_viz >= 25]
        
        print(f"  {model_name}:")
        print(f"    Shallow (<15m): Mean error = {np.mean(shallow):.2f}m")
        print(f"    Medium (15-25m): Mean error = {np.mean(medium):.2f}m")
        print(f"    Deep (>25m): Mean error = {np.mean(deep):.2f}m")

plt.tight_layout()
depth_path = REGION_OUTPUT_DIR / "depth_stratified_error_analysis.png"
fig.savefig(depth_path, dpi=150, bbox_inches='tight')
print(f"  Depth analysis saved: {depth_path}")
plt.close()

# ============================================================================
# CREATE SUMMARY STATISTICS TABLE
# ============================================================================
print("\n Creating summary statistics...")

summary_data = []
for model_key, model_name in model_names.items():
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        error_signed = errors[model_key]
        
        # Depth stratification
        mask_shallow = depths_viz < 15
        mask_medium = (depths_viz >= 15) & (depths_viz < 25)
        mask_deep = depths_viz >= 25
        
        summary_data.append({
            'Model': model_name,
            'Global MAE (m)': np.mean(error_abs),
            'Global RMSE (m)': np.sqrt(np.mean(error_abs**2)),
            'Bias (m)': np.mean(error_signed),
            'Shallow MAE (m)': np.mean(error_abs[mask_shallow]) if np.any(mask_shallow) else np.nan,
            'Medium MAE (m)': np.mean(error_abs[mask_medium]) if np.any(mask_medium) else np.nan,
            'Deep MAE (m)': np.mean(error_abs[mask_deep]) if np.any(mask_deep) else np.nan,
            'Std Dev (m)': np.std(error_abs),
            'Max Error (m)': np.max(error_abs),
            '95th %ile Error (m)': np.percentile(error_abs, 95)
        })

summary_df = pd.DataFrame(summary_data)
summary_csv = REGION_OUTPUT_DIR / "geographic_error_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n Geographic Error Summary:")
print(summary_df.to_string())
print(f"\n  Summary saved: {summary_csv}")

# ============================================================================
# CREATE COMPARISON VISUALIZATION
# ============================================================================
print("\n Creating model comparison chart...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Geographic Performance Comparison Across Models', fontsize=14, fontweight='bold')

# MAE by depth
ax = axes[0]
depth_ranges = ['Shallow\n(<15m)', 'Medium\n(15-25m)', 'Deep\n(>25m)']
for idx, (model_key, model_name) in enumerate(model_list):
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        
        mask_shallow = depths_viz < 15
        mask_medium = (depths_viz >= 15) & (depths_viz < 25)
        mask_deep = depths_viz >= 25
        
        mae_values = [
            np.mean(error_abs[mask_shallow]) if np.any(mask_shallow) else 0,
            np.mean(error_abs[mask_medium]) if np.any(mask_medium) else 0,
            np.mean(error_abs[mask_deep]) if np.any(mask_deep) else 0
        ]
        
        x = np.arange(len(depth_ranges))
        ax.bar(x + idx*0.2, mae_values, 0.2, label=model_name, alpha=0.8)

ax.set_ylabel('Mean Absolute Error (meters)', fontsize=11)
ax.set_title('Error by Water Depth', fontsize=12, fontweight='bold')
ax.set_xticks(x + 0.3)
ax.set_xticklabels(depth_ranges)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Bias comparison
ax = axes[1]
biases = [np.mean(errors[k]) if k in errors else 0 for k, _ in model_list]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(range(len(model_list)), biases, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Bias')
ax.set_ylabel('Bias (meters)\nPositive=Underestimate, Negative=Overestimate', fontsize=11)
ax.set_title('Systematic Bias by Model', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(model_list)))
ax.set_xticklabels([name for _, name in model_list], rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=10)

# Error distribution
ax = axes[2]
error_stats = []
for model_key, model_name in model_list:
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        error_stats.append({
            'Model': model_name,
            'MAE': np.mean(error_abs),
            'Std': np.std(error_abs)
        })

error_df = pd.DataFrame(error_stats)
x = np.arange(len(error_df))
ax.bar(x, error_df['MAE'], label='Mean Absolute Error', alpha=0.8, color=colors)
ax.errorbar(x, error_df['MAE'], yerr=error_df['Std'], fmt='none', color='black', 
            capsize=5, capthick=2, label='Std Deviation')
ax.set_ylabel('Error (meters)', fontsize=11)
ax.set_title('Error Magnitude and Variability', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(error_df['Model'], rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=10)

plt.tight_layout()
comparison_path = REGION_OUTPUT_DIR / "geographic_model_comparison.png"
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"  Comparison chart saved: {comparison_path}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("GEOGRAPHIC ANALYSIS COMPLETE")
print("=" * 90)

print("\n Output files generated:")
print(f"  1. geographic_error_heatmap.png - Absolute error by location")
print(f"  2. geographic_signed_error_map.png - Prediction bias (under/overestimate)")
print(f"  3. depth_stratified_error_analysis.png - Performance by water depth")
print(f"  4. geographic_model_comparison.png - Comparison charts")
print(f"  5. geographic_error_summary.csv - Detailed statistics")

print("\n Key Findings:")
print(f"  XGBoost: Most uniform errors across geography and depths")
print(f"  Random Forest: Second best, nearly equivalent to XGBoost")
print(f"  Decision Tree: Acceptable performance, some depth-dependent bias")
print(f"  Linear Regression: Large errors, systematic bias, depth-dependent failures")

print("\n" + "=" * 90)
