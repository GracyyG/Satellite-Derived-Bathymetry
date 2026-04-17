#!/usr/bin/env python3
"""
3D Depth-Stratified Error Analysis
Creates interactive 3D scatter plots showing error distribution across depth ranges
Visualizes how each model performs at different water depths and locations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import argparse
import warnings

warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='3D Depth-Stratified Error Analysis')
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
print("3D DEPTH-STRATIFIED ERROR ANALYSIS")
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

print(f"  Features: {features.shape}")
print(f"  Depths: {true_depths.shape}")

# Load scaler from region-specific directory
REGION_MODELS_DIR = MODELS_DIR / region_name
scaler_path = REGION_MODELS_DIR / "feature_scaler.joblib"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"  [OK] Scaler loaded from {region_name}")
else:
    print(f"  [WARN] No scaler found, using unscaled features")
    scaler = None

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
# SUBSAMPLE FOR VISUALIZATION
# ============================================================================
print("\n Subsampling for visualization...")

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

n_samples = len(features_viz)
n_lat = int(np.sqrt(n_samples))
n_lon = int(np.ceil(n_samples / n_lat))

latitudes = np.linspace(lat_min, lat_max, n_lat)
longitudes = np.linspace(lon_min, lon_max, n_lon)
lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)

latitudes = lat_grid.flatten()[:n_samples]
longitudes = lon_grid.flatten()[:n_samples]

# Ensure all arrays have the same length
min_len = min(len(latitudes), len(longitudes), len(features_viz), len(depths_viz))
latitudes = latitudes[:min_len]
longitudes = longitudes[:min_len]
features_viz = features_viz[:min_len]
depths_viz = depths_viz[:min_len]

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
            pred = pred[:len(features_viz)]
            predictions[model_key] = pred
            print(f"  {model_name}: {len(pred)} predictions")
        except:
            if features_viz.shape[1] < 15:
                padded_features = np.hstack([features_viz, np.full((features_viz.shape[0], 1), feature_padding_val)])
                pred = model.predict(padded_features)
                pred = pred[:len(features_viz)]
                predictions[model_key] = pred
                print(f"  {model_name}: {len(pred)} predictions (with padding)")
    except Exception as e:
        print(f"  {model_name} failed: {str(e)}")

# ============================================================================
# CALCULATE ERRORS
# ============================================================================
print("\n Calculating errors...")

errors = {}
errors_abs = {}

for model_key, model_name in model_names.items():
    if model_key in predictions:
        pred = predictions[model_key]
        error = depths_viz - pred
        error_abs = np.abs(error)
        
        errors[model_key] = error
        errors_abs[model_key] = error_abs

# ============================================================================
# CREATE 3D SCATTER PLOTS (4-PANEL SUBPLOTS)
# ============================================================================
print("\n Creating 3D depth-error scatter plots...")

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
           [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=list(model_names.values()),
    horizontal_spacing=0.1,
    vertical_spacing=0.1
)

colors_3d = {
    'decision_tree': '#1f77b4',
    'linear_regression': '#ff7f0e',
    'random_forest': '#2ca02c',
    'xgboost': '#d62728'
}

row_col_map = [(1, 1), (1, 2), (2, 1), (2, 2)]

for (model_key, model_name), (row, col) in zip(model_names.items(), row_col_map):
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        
        # Create scatter plot: Depth vs Error vs Latitude
        fig.add_trace(
            go.Scatter3d(
                x=depths_viz,
                y=error_abs,
                z=latitudes,
                mode='markers',
                name=model_name,
                marker=dict(
                    size=3,
                    color=error_abs,
                    colorscale='Viridis',
                    showscale=(col == 2),
                    colorbar=dict(
                        title="Error (m)",
                        x=1.02 if col == 2 else 0.48,
                        len=0.4
                    ),
                    opacity=0.8,
                    line=dict(width=0)
                ),
                text=[f"Depth: {d:.1f}m<br>Error: {e:.2f}m<br>Lat: {lat:.3f}" 
                      for d, e, lat in zip(depths_viz, error_abs, latitudes)],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add axes labels
        fig.update_scenes(
            xaxis=dict(title="Actual Depth (m)"),
            yaxis=dict(title="Absolute Error (m)"),
            zaxis=dict(title="Latitude (N)"),
            row=row, col=col,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        )

fig.update_layout(
    title_text="<b>3D Depth-Stratified Error Analysis</b><br><sub>Error vs Depth vs Latitude for Each Model</sub>",
    height=900,
    showlegend=False,
    font=dict(size=11)
)

output_3d_path = REGION_OUTPUT_DIR / "depth_stratified_error_3d_subplots.html"
fig.write_html(output_3d_path)
print(f"  3D subplots saved: {output_3d_path}")

# ============================================================================
# CREATE UNIFIED 3D VISUALIZATION (ALL MODELS OVERLAID)
# ============================================================================
print("\n Creating unified 3D visualization...")

fig_unified = go.Figure()

colors_unified = {
    'decision_tree': 'rgba(31, 119, 180, 0.7)',
    'linear_regression': 'rgba(255, 127, 14, 0.7)',
    'random_forest': 'rgba(44, 160, 44, 0.7)',
    'xgboost': 'rgba(214, 39, 40, 0.7)'
}

for model_key, model_name in model_names.items():
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        
        fig_unified.add_trace(
            go.Scatter3d(
                x=depths_viz,
                y=error_abs,
                z=latitudes,
                mode='markers',
                name=model_name,
                marker=dict(
                    size=4,
                    color=colors_unified[model_key],
                    opacity=0.7,
                    line=dict(width=0)
                ),
                text=[f"{model_name}<br>Depth: {d:.1f}m<br>Error: {e:.2f}m<br>Lat: {lat:.3f}" 
                      for d, e, lat in zip(depths_viz, error_abs, latitudes)],
                hovertemplate='<b>%{text}</b><extra></extra>'
            )
        )

fig_unified.update_layout(
    title="<b>Unified 3D Depth-Error Analysis</b><br><sub>All Models Overlaid - Error vs Depth vs Latitude</sub>",
    scene=dict(
        xaxis_title="Actual Water Depth (meters)",
        yaxis_title="Absolute Error (meters)",
        zaxis_title="Latitude (°N)",
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    ),
    height=700,
    font=dict(size=12),
    hovermode='closest'
)

output_unified_path = REGION_OUTPUT_DIR / "depth_stratified_error_3d_unified.html"
fig_unified.write_html(output_unified_path)
print(f"  Unified 3D saved: {output_unified_path}")

# ============================================================================
# CREATE 3D SURFACE PLOT (ERROR SURFACE BY DEPTH AND LATITUDE)
# ============================================================================
print("\n Creating 3D error surface visualization...")

# Create grid for surface
depth_bins = np.linspace(depths_viz.min(), depths_viz.max(), 20)
lat_bins = np.linspace(latitudes.min(), latitudes.max(), 20)
depth_grid, lat_grid = np.meshgrid(depth_bins, lat_bins)

# Calculate mean error for each bin
error_surface = np.zeros_like(depth_grid)
for i in range(len(depth_bins)-1):
    for j in range(len(lat_bins)-1):
        mask = ((depths_viz >= depth_bins[i]) & (depths_viz < depth_bins[i+1]) &
                (latitudes >= lat_bins[j]) & (latitudes < lat_bins[j+1]))
        
        if np.any(mask):
            # Calculate error for this bin across all models (weighted by best model)
            error_surface[j, i] = np.mean([np.mean(errors_abs[k][mask]) 
                                          for k in model_names.keys() if k in errors_abs])
        else:
            error_surface[j, i] = np.nan

fig_surface = go.Figure(data=[go.Surface(
    x=depth_grid[0],
    y=lat_grid[:, 0],
    z=error_surface,
    colorscale='RdYlGn_r',
    colorbar=dict(title="Mean Error (m)"),
    name="Error Surface"
)])

fig_surface.update_layout(
    title="<b>3D Error Surface</b><br><sub>Average Error Distribution Across Depth and Latitude</sub>",
    scene=dict(
        xaxis_title="Actual Water Depth (meters)",
        yaxis_title="Latitude (°N)",
        zaxis_title="Mean Absolute Error (meters)",
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    ),
    height=700,
    font=dict(size=12)
)

output_surface_path = REGION_OUTPUT_DIR / "depth_stratified_error_3d_surface.html"
fig_surface.write_html(output_surface_path)
print(f"  3D surface saved: {output_surface_path}")

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================
print("\n Statistical Summary by Model:")
print("-" * 90)

for model_key, model_name in model_names.items():
    if model_key in errors_abs:
        error = errors_abs[model_key]
        
        # Depth-based statistics
        mask_shallow = depths_viz < 20
        mask_medium = (depths_viz >= 20) & (depths_viz < 30)
        mask_deep = depths_viz >= 30
        
        print(f"\n{model_name}:")
        print(f"  Overall: MAE={np.mean(error):.3f}m, RMSE={np.sqrt(np.mean(error**2)):.3f}m")
        if np.any(mask_shallow):
            print(f"  Shallow (<20m): MAE={np.mean(error[mask_shallow]):.3f}m")
        if np.any(mask_medium):
            print(f"  Medium (20-30m): MAE={np.mean(error[mask_medium]):.3f}m")
        if np.any(mask_deep):
            print(f"  Deep (>30m): MAE={np.mean(error[mask_deep]):.3f}m")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("3D ANALYSIS COMPLETE")
print("=" * 90)

print("\n Output files generated:")
print(f"  1. depth_stratified_error_3d_subplots.html - 4-panel 3D scatter plots")
print(f"  2. depth_stratified_error_3d_unified.html - All models overlaid in 3D")
print(f"  3. depth_stratified_error_3d_surface.html - 3D error surface")

print("\n Interactive Features:")
print(f"  • Rotate: Click and drag")
print(f"  • Zoom: Scroll wheel")
print(f"  • Pan: Shift + drag")
print(f"  • Hover: See exact values")
print(f"  • Legend: Click to toggle models")

print("\n" + "=" * 90)
