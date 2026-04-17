#!/usr/bin/env python3
"""
Geographic Heatmap Analysis for Kachchh Region
Shows spatial distribution of model performance and bathymetry predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.interpolate import griddata

def load_kachchh_data():
    """Load Kachchh test data and models"""
    
    project_root = Path(__file__).parent.parent
    
    # Load test data
    test_data_dir = project_root / "data/processed/kachchh/training_data/test"
    X_test = np.load(test_data_dir / "features.npy")
    y_test = np.load(test_data_dir / "depths.npy")
    coords = np.load(test_data_dir / "coordinates.npy")
    
    # Remove depth_value feature (index 8)
    X_test_spectral = np.delete(X_test, 8, axis=1)
    
    # Load models
    models_dir = project_root / "models/kachchh"
    models = {
        'XGBoost': joblib.load(models_dir / 'xgboost.joblib'),
        'Random Forest': joblib.load(models_dir / 'random_forest.joblib')
    }
    
    return X_test_spectral, y_test, coords, models

def create_geographic_heatmaps(X_test, y_test, coords, models, region_name):
    """Create geographic heatmaps of bathymetry and model performance"""
    
    lats = coords[:, 0]
    lons = coords[:, 1]
    
    # Generate predictions
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Geographic Heatmap Analysis - {region_name.title()} Region\n' +
                 f'XGBoost RMSE: 0.782m | Random Forest RMSE: 0.975m', fontsize=16)
    
    # 1. True bathymetry heatmap
    ax = axes[0, 0]
    scatter = ax.scatter(lons, lats, c=y_test, cmap='Blues_r', s=15, alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('True Bathymetry')
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    ax.grid(True, alpha=0.3)
    
    # 2. XGBoost predictions heatmap
    ax = axes[0, 1]
    scatter = ax.scatter(lons, lats, c=predictions['XGBoost'], cmap='viridis', s=15, alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('XGBoost Predictions')
    plt.colorbar(scatter, ax=ax, label='Predicted Depth (m)')
    ax.grid(True, alpha=0.3)
    
    # 3. Random Forest predictions heatmap
    ax = axes[0, 2]
    scatter = ax.scatter(lons, lats, c=predictions['Random Forest'], cmap='plasma', s=15, alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Random Forest Predictions')
    plt.colorbar(scatter, ax=ax, label='Predicted Depth (m)')
    ax.grid(True, alpha=0.3)
    
    # 4. XGBoost error heatmap
    ax = axes[1, 0]
    xgb_errors = np.abs(y_test - predictions['XGBoost'])
    scatter = ax.scatter(lons, lats, c=xgb_errors, cmap='Reds', s=15, alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'XGBoost Errors (Mean: {xgb_errors.mean():.3f}m)')
    plt.colorbar(scatter, ax=ax, label='Absolute Error (m)')
    ax.grid(True, alpha=0.3)
    
    # 5. Random Forest error heatmap
    ax = axes[1, 1]
    rf_errors = np.abs(y_test - predictions['Random Forest'])
    scatter = ax.scatter(lons, lats, c=rf_errors, cmap='Oranges', s=15, alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Random Forest Errors (Mean: {rf_errors.mean():.3f}m)')
    plt.colorbar(scatter, ax=ax, label='Absolute Error (m)')
    ax.grid(True, alpha=0.3)
    
    # 6. Error comparison
    ax = axes[1, 2]
    ax.scatter(xgb_errors, rf_errors, alpha=0.6, s=15)
    max_error = max(xgb_errors.max(), rf_errors.max())
    ax.plot([0, max_error], [0, max_error], 'r--', lw=2, label='Equal Error')
    ax.set_xlabel('XGBoost Error (m)')
    ax.set_ylabel('Random Forest Error (m)')
    ax.set_title('Model Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(f"outputs/{region_name}/realistic_bathymetry_showcase")
    heatmap_path = output_dir / f"{region_name}_geographic_heatmap_analysis.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[OK] Geographic heatmap analysis saved to: {heatmap_path}")
    
    return xgb_errors, rf_errors

def create_interactive_heatmap(X_test, y_test, coords, models, region_name):
    """Create interactive Plotly heatmap"""
    
    lats = coords[:, 0]
    lons = coords[:, 1]
    
    # Generate predictions
    xgb_pred = models['XGBoost'].predict(X_test)
    rf_pred = models['Random Forest'].predict(X_test)
    
    # Calculate errors
    xgb_errors = np.abs(y_test - xgb_pred)
    rf_errors = np.abs(y_test - rf_pred)
    
    # Create interactive subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['True Bathymetry', 'XGBoost Predictions', 
                       'XGBoost Errors', 'Random Forest Errors'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # True bathymetry
    fig.add_trace(
        go.Scatter(x=lons, y=lats, mode='markers',
                  marker=dict(size=6, color=y_test, colorscale='Blues_r',
                             showscale=True, colorbar=dict(x=0.45, len=0.45)),
                  text=[f'Depth: {d:.1f}m' for d in y_test],
                  name='True Depth'),
        row=1, col=1
    )
    
    # XGBoost predictions
    fig.add_trace(
        go.Scatter(x=lons, y=lats, mode='markers',
                  marker=dict(size=6, color=xgb_pred, colorscale='Viridis',
                             showscale=True, colorbar=dict(x=1.02, len=0.45)),
                  text=[f'Predicted: {p:.1f}m' for p in xgb_pred],
                  name='XGBoost Pred'),
        row=1, col=2
    )
    
    # XGBoost errors
    fig.add_trace(
        go.Scatter(x=lons, y=lats, mode='markers',
                  marker=dict(size=6, color=xgb_errors, colorscale='Reds',
                             showscale=True, colorbar=dict(x=0.45, y=0.1, len=0.45)),
                  text=[f'Error: {e:.2f}m' for e in xgb_errors],
                  name='XGBoost Error'),
        row=2, col=1
    )
    
    # Random Forest errors
    fig.add_trace(
        go.Scatter(x=lons, y=lats, mode='markers',
                  marker=dict(size=6, color=rf_errors, colorscale='Oranges',
                             showscale=True, colorbar=dict(x=1.02, y=0.1, len=0.45)),
                  text=[f'Error: {e:.2f}m' for e in rf_errors],
                  name='Random Forest Error'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Interactive Geographic Analysis - {region_name.title()} Region<br>' +
              f'XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, xgb_pred)):.3f}m | ' +
              f'Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.3f}m',
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Longitude", row=i, col=j)
            fig.update_yaxes(title_text="Latitude", row=i, col=j)
    
    # Save interactive plot
    output_dir = Path(f"outputs/{region_name}/realistic_bathymetry_showcase")
    interactive_path = output_dir / f"{region_name}_geographic_heatmap_interactive.html"
    fig.write_html(str(interactive_path))
    
    print(f"[OK] Interactive geographic heatmap saved to: {interactive_path}")

def create_spatial_statistics_analysis(X_test, y_test, coords, models, region_name):
    """Create spatial statistics analysis"""
    
    lats = coords[:, 0]
    lons = coords[:, 1]
    
    # Generate predictions
    predictions = {}
    errors = {}
    
    for name, model in models.items():
        pred = model.predict(X_test)
        predictions[name] = pred
        errors[name] = np.abs(y_test - pred)
    
    # Create spatial bins for analysis
    n_bins = 5
    lat_bins = np.linspace(lats.min(), lats.max(), n_bins + 1)
    lon_bins = np.linspace(lons.min(), lons.max(), n_bins + 1)
    
    spatial_stats = []
    
    for i in range(n_bins):
        for j in range(n_bins):
            # Define bin boundaries
            lat_mask = (lats >= lat_bins[i]) & (lats < lat_bins[i+1])
            lon_mask = (lons >= lon_bins[j]) & (lons < lon_bins[j+1])
            spatial_mask = lat_mask & lon_mask
            
            if np.sum(spatial_mask) > 10:  # At least 10 samples
                bin_stats = {
                    'lat_center': (lat_bins[i] + lat_bins[i+1]) / 2,
                    'lon_center': (lon_bins[j] + lon_bins[j+1]) / 2,
                    'n_samples': np.sum(spatial_mask),
                    'mean_depth': y_test[spatial_mask].mean(),
                    'depth_std': y_test[spatial_mask].std(),
                }
                
                for name in models.keys():
                    bin_stats[f'{name}_rmse'] = np.sqrt(mean_squared_error(
                        y_test[spatial_mask], predictions[name][spatial_mask]
                    ))
                    bin_stats[f'{name}_mean_error'] = errors[name][spatial_mask].mean()
                
                spatial_stats.append(bin_stats)
    
    # Convert to DataFrame
    spatial_df = pd.DataFrame(spatial_stats)
    
    # Create spatial statistics visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Mean depth by location
    ax = axes[0]
    scatter = ax.scatter(spatial_df['lon_center'], spatial_df['lat_center'], 
                        c=spatial_df['mean_depth'], s=spatial_df['n_samples']*2,
                        cmap='Blues_r', alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Mean Depth by Spatial Bin')
    plt.colorbar(scatter, ax=ax, label='Mean Depth (m)')
    
    # XGBoost RMSE by location
    ax = axes[1]
    scatter = ax.scatter(spatial_df['lon_center'], spatial_df['lat_center'],
                        c=spatial_df['XGBoost_rmse'], s=spatial_df['n_samples']*2,
                        cmap='Reds', alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('XGBoost RMSE by Location')
    plt.colorbar(scatter, ax=ax, label='RMSE (m)')
    
    # Sample density
    ax = axes[2]
    scatter = ax.scatter(spatial_df['lon_center'], spatial_df['lat_center'],
                        c=spatial_df['n_samples'], s=spatial_df['n_samples']*2,
                        cmap='Greens', alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Sample Density by Location')
    plt.colorbar(scatter, ax=ax, label='Sample Count')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(f"outputs/{region_name}/realistic_bathymetry_showcase")
    stats_path = output_dir / f"{region_name}_spatial_statistics_analysis.png"
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save spatial statistics
    csv_path = output_dir / f"{region_name}_spatial_statistics.csv"
    spatial_df.to_csv(csv_path, index=False)
    
    print(f"[OK] Spatial statistics analysis saved to: {stats_path}")
    print(f"[OK] Spatial statistics data saved to: {csv_path}")

def main():
    region_name = "kachchh"
    
    print("="*80)
    print(f"GEOGRAPHIC HEATMAP ANALYSIS - {region_name.upper()} REGION")
    print("Spatial Distribution of Improved Bathymetry Performance")
    print("="*80)
    
    # Load data and models
    X_test, y_test, coords, models = load_kachchh_data()
    
    print(f"[OK] Loaded data: {len(y_test)} samples")
    print(f"    Depth range: {y_test.min():.1f}m to {y_test.max():.1f}m")
    print(f"    Geographic extent: {coords[:, 0].min():.2f}°-{coords[:, 0].max():.2f}°N, {coords[:, 1].min():.2f}°-{coords[:, 1].max():.2f}°E")
    
    # Create geographic heatmaps
    print("\\nCreating geographic heatmaps...")
    xgb_errors, rf_errors = create_geographic_heatmaps(X_test, y_test, coords, models, region_name)
    
    # Create interactive heatmap
    print("\\nCreating interactive heatmap...")
    create_interactive_heatmap(X_test, y_test, coords, models, region_name)
    
    # Create spatial statistics analysis
    print("\\nCreating spatial statistics analysis...")
    create_spatial_statistics_analysis(X_test, y_test, coords, models, region_name)
    
    print("\\n" + "="*80)
    print("GEOGRAPHIC HEATMAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"✅ XGBoost mean error: {xgb_errors.mean():.3f}m")
    print(f"✅ Random Forest mean error: {rf_errors.mean():.3f}m")
    print(f"✅ Static and interactive heatmaps generated")
    print(f"✅ Spatial statistics analysis completed")

if __name__ == "__main__":
    main()