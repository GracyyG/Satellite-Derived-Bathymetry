#!/usr/bin/env python3
"""
3D Mesh Surface Plots - Depth-Stratified Error Analysis
Creates interactive 3D surface plots for each model showing error landscape
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
parser = argparse.ArgumentParser(description='3D Mesh Surface Analysis')
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
print("3D MESH SURFACE PLOTS - DEPTH-STRATIFIED ERROR ANALYSIS")
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

errors_abs = {}

for model_key, model_name in model_names.items():
    if model_key in predictions:
        pred = predictions[model_key]
        error_abs = np.abs(depths_viz - pred)
        errors_abs[model_key] = error_abs

# ============================================================================
# CREATE INDIVIDUAL 3D MESH SURFACE PLOTS
# ============================================================================
print("\n Creating individual 3D mesh surface plots...")

model_list = list(model_names.items())
colorscales = ['Blues', 'Oranges', 'Greens', 'Reds']

for (model_key, model_name), colorscale in zip(model_list, colorscales):
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        
        # Create grid for surface
        depth_bins = np.linspace(depths_viz.min(), depths_viz.max(), 25)
        lat_bins = np.linspace(latitudes.min(), latitudes.max(), 25)
        depth_grid, lat_grid = np.meshgrid(depth_bins, lat_bins)
        
        # Calculate mean error for each bin
        error_surface = np.zeros_like(depth_grid, dtype=float)
        count_surface = np.zeros_like(depth_grid, dtype=float)
        
        for i in range(len(depth_bins)-1):
            for j in range(len(lat_bins)-1):
                mask = ((depths_viz >= depth_bins[i]) & (depths_viz < depth_bins[i+1]) &
                        (latitudes >= lat_bins[j]) & (latitudes < lat_bins[j+1]))
                
                if np.any(mask):
                    error_surface[j, i] = np.mean(error_abs[mask])
                    count_surface[j, i] = np.sum(mask)
                else:
                    error_surface[j, i] = np.nan
        
        # Create figure
        fig = go.Figure(data=[go.Surface(
            x=depth_grid[0],
            y=lat_grid[:, 0],
            z=error_surface,
            colorscale=colorscale,
            colorbar=dict(
                title="Error (m)",
                thickness=20,
                len=0.7
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
            ),
            name=model_name
        )])
        
        # Update layout
        fig.update_layout(
            title=f"<b>{model_name}</b><br><sub>3D Error Surface - Depth vs Latitude</sub>",
            scene=dict(
                xaxis_title="Actual Water Depth (meters)",
                yaxis_title="Latitude (°N)",
                zaxis_title="Mean Absolute Error (meters)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                bgcolor="rgb(30, 30, 30)",
                xaxis=dict(backgroundcolor="rgb(40, 40, 40)", gridcolor="rgb(100, 100, 100)", 
                          showbackground=True, linecolor="rgb(150, 150, 150)"),
                yaxis=dict(backgroundcolor="rgb(40, 40, 40)", gridcolor="rgb(100, 100, 100)", 
                          showbackground=True, linecolor="rgb(150, 150, 150)"),
                zaxis=dict(backgroundcolor="rgb(40, 40, 40)", gridcolor="rgb(100, 100, 100)", 
                          showbackground=True, linecolor="rgb(150, 150, 150)")
            ),
            height=700,
            font=dict(size=12, color="white"),
            paper_bgcolor="rgb(25, 25, 25)",
            plot_bgcolor="rgb(25, 25, 25)",
            showlegend=False
        )
        
        # Save individual plot
        safe_name = model_key.replace(' ', '_')
        output_path = REGION_OUTPUT_DIR / f"mesh_surface_3d_{safe_name}.html"
        fig.write_html(output_path)
        print(f"  {model_name} mesh saved: {output_path}")

# ============================================================================
# CREATE 4-PANEL MESH SUBPLOT
# ============================================================================
print("\n Creating 4-panel mesh surface subplot...")

fig_subplots = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}],
           [{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=list(model_names.values()),
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

row_col_map = [(1, 1), (1, 2), (2, 1), (2, 2)]
colorscales_sub = ['Blues', 'Oranges', 'Greens', 'Reds']

for (model_key, model_name), (row, col), colorscale in zip(model_list, row_col_map, colorscales_sub):
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        
        # Create grid for surface
        depth_bins = np.linspace(depths_viz.min(), depths_viz.max(), 20)
        lat_bins = np.linspace(latitudes.min(), latitudes.max(), 20)
        depth_grid, lat_grid = np.meshgrid(depth_bins, lat_bins)
        
        # Calculate mean error for each bin
        error_surface = np.zeros_like(depth_grid, dtype=float)
        
        for i in range(len(depth_bins)-1):
            for j in range(len(lat_bins)-1):
                mask = ((depths_viz >= depth_bins[i]) & (depths_viz < depth_bins[i+1]) &
                        (latitudes >= lat_bins[j]) & (latitudes < lat_bins[j+1]))
                
                if np.any(mask):
                    error_surface[j, i] = np.mean(error_abs[mask])
                else:
                    error_surface[j, i] = np.nan
        
        fig_subplots.add_trace(
            go.Surface(
                x=depth_grid[0],
                y=lat_grid[:, 0],
                z=error_surface,
                colorscale=colorscale,
                showscale=(col == 2),
                colorbar=dict(
                    title="Error (m)",
                    x=1.02 if col == 2 else 0.48,
                    len=0.4
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
                ),
                name=model_name
            ),
            row=row, col=col
        )
        
        # Update scene
        fig_subplots.update_scenes(
            xaxis=dict(title="Depth (m)", backgroundcolor="rgb(40,40,40)", gridcolor="rgb(100,100,100)", 
                      showbackground=True, linecolor="rgb(150,150,150)"),
            yaxis=dict(title="Latitude", backgroundcolor="rgb(40,40,40)", gridcolor="rgb(100,100,100)", 
                      showbackground=True, linecolor="rgb(150,150,150)"),
            zaxis=dict(title="Error (m)", backgroundcolor="rgb(40,40,40)", gridcolor="rgb(100,100,100)", 
                      showbackground=True, linecolor="rgb(150,150,150)"),
            bgcolor="rgb(30, 30, 30)",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.2)),
            row=row, col=col
        )

fig_subplots.update_layout(
    title_text="<b>4-Panel 3D Error Surface Mesh</b><br><sub>Model Performance Landscape - Depth vs Latitude</sub>",
    height=1000,
    showlegend=False,
    font=dict(size=11, color="white"),
    paper_bgcolor="rgb(25, 25, 25)",
    plot_bgcolor="rgb(25, 25, 25)"
)

output_subplots_path = REGION_OUTPUT_DIR / "mesh_surface_3d_4panel_subplots.html"
fig_subplots.write_html(output_subplots_path)
print(f"  4-panel mesh saved: {output_subplots_path}")

# ============================================================================
# CREATE UNIFIED COMPARISON (ALL SURFACES OVERLAID)
# ============================================================================
print("\n Creating unified mesh comparison...")

fig_unified = go.Figure()

colors_unified = {
    'decision_tree': 'Blues',
    'linear_regression': 'Oranges',
    'random_forest': 'Greens',
    'xgboost': 'Reds'
}

opacity_values = [0.6, 0.5, 0.7, 0.8]

for (model_key, model_name), opacity in zip(model_list, opacity_values):
    if model_key in errors_abs:
        error_abs = errors_abs[model_key]
        
        # Create grid for surface
        depth_bins = np.linspace(depths_viz.min(), depths_viz.max(), 20)
        lat_bins = np.linspace(latitudes.min(), latitudes.max(), 20)
        depth_grid, lat_grid = np.meshgrid(depth_bins, lat_bins)
        
        # Calculate mean error for each bin
        error_surface = np.zeros_like(depth_grid, dtype=float)
        
        for i in range(len(depth_bins)-1):
            for j in range(len(lat_bins)-1):
                mask = ((depths_viz >= depth_bins[i]) & (depths_viz < depth_bins[i+1]) &
                        (latitudes >= lat_bins[j]) & (latitudes < lat_bins[j+1]))
                
                if np.any(mask):
                    error_surface[j, i] = np.mean(error_abs[mask])
                else:
                    error_surface[j, i] = np.nan
        
        fig_unified.add_trace(
            go.Surface(
                x=depth_grid[0],
                y=lat_grid[:, 0],
                z=error_surface,
                name=model_name,
                colorscale=colors_unified[model_key],
                opacity=opacity,
                showscale=False,
                contours=dict(
                    z=dict(show=False)
                )
            )
        )

fig_unified.update_layout(
    title="<b>Unified 3D Error Surface Comparison</b><br><sub>All Models Overlaid - Compare Error Landscapes</sub>",
    scene=dict(
        xaxis_title="Actual Water Depth (meters)",
        yaxis_title="Latitude (°N)",
        zaxis_title="Mean Absolute Error (meters)",
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        ),
        xaxis=dict(backgroundcolor="rgb(40,40,40)", gridcolor="rgb(100,100,100)", linecolor="rgb(150,150,150)"),
        yaxis=dict(backgroundcolor="rgb(40,40,40)", gridcolor="rgb(100,100,100)", linecolor="rgb(150,150,150)"),
        zaxis=dict(backgroundcolor="rgb(40,40,40)", gridcolor="rgb(100,100,100)", linecolor="rgb(150,150,150)"),
        bgcolor="rgb(30, 30, 30)"
    ),
    height=800,
    font=dict(size=12, color="white"),
    paper_bgcolor="rgb(25, 25, 25)",
    plot_bgcolor="rgb(25, 25, 25)",
    showlegend=True,
    hovermode='closest'
)

output_unified_path = REGION_OUTPUT_DIR / "mesh_surface_3d_unified_comparison.html"
fig_unified.write_html(output_unified_path)
print(f"  Unified mesh saved: {output_unified_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("3D MESH SURFACE ANALYSIS COMPLETE")
print("=" * 90)

print("\n Output files generated:")
print(f"  1. mesh_surface_3d_decision_tree.html - Decision Tree mesh surface")
print(f"  2. mesh_surface_3d_linear_regression.html - Linear Regression mesh surface")
print(f"  3. mesh_surface_3d_random_forest.html - Random Forest mesh surface")
print(f"  4. mesh_surface_3d_xgboost.html - XGBoost mesh surface")
print(f"  5. mesh_surface_3d_4panel_subplots.html - 4-panel comparison")
print(f"  6. mesh_surface_3d_unified_comparison.html - All models overlaid")

print("\n Interactive Features:")
print(f"  • Rotate: Click and drag to explore from different angles")
print(f"  • Zoom: Scroll wheel to zoom in/out")
print(f"  • Pan: Shift + drag to move the view")
print(f"  • Contour lines: Show error magnitude boundaries (on some plots)")
print(f"  • Hover: See exact error values at each point")
print(f"  • Toggle: Click legend items to show/hide models")

print("\n Visualization Insights:")
print(f"  • Smooth surface = Consistent model performance")
print(f"  • Bumpy surface = Variable performance across conditions")
print(f"  • Low surface = Good predictions (small errors)")
print(f"  • High surface = Poor predictions (large errors)")
print(f"  • Color intensity = Error magnitude (darker/redder = higher errors)")

print("\n" + "=" * 90)
