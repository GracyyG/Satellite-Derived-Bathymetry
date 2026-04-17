#!/usr/bin/env python3
"""
Generate comprehensive visualizations for Palk Strait region
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_palk_strait_data():
    """Load training data and models for palk_strait"""
    
    # Paths
    data_dir = Path("data/sentinel/palk_strait/processed/training_data")
    models_dir = Path("models/palk_strait")
    output_dir = Path("outputs/palk_strait/final_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌊 Loading Palk Strait Data...")
    
    # Load training data
    features = np.load(data_dir / "features.npy")
    depths = np.load(data_dir / "depths.npy") 
    scaler = joblib.load(data_dir / "feature_scaler.joblib")
    
    # Load metadata
    with open(data_dir / "feature_metadata.json") as f:
        metadata = json.load(f)
    
    print(f"   Features shape: {features.shape}")
    print(f"   Depths shape: {depths.shape}")
    print(f"   Depth range: {depths.min():.2f}m to {depths.max():.2f}m")
    
    # Load models
    models = {}
    for model_file in models_dir.glob("*.joblib"):
        if "scaler" not in model_file.name:
            model_name = model_file.stem.replace('_', ' ').title()
            models[model_name] = joblib.load(model_file)
            print(f"   Loaded: {model_name}")
    
    return features, depths, scaler, models, metadata, output_dir

def generate_model_predictions(features, depths, scaler, models):
    """Generate predictions from all models"""
    
    print("\n🔮 Generating Model Predictions...")
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Sample data for visualization (every 100th point)
    sample_indices = np.arange(0, len(features), 100)
    features_sample = features_scaled[sample_indices]
    depths_sample = depths[sample_indices]
    
    print(f"   Sampled {len(sample_indices)} points for visualization")
    
    # Generate predictions
    predictions = {}
    metrics = {}
    
    for name, model in models.items():
        pred = model.predict(features_sample)
        predictions[name] = pred
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(depths_sample, pred))
        mae = mean_absolute_error(depths_sample, pred)
        r2 = r2_score(depths_sample, pred)
        
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
        print(f"   {name}: RMSE={rmse:.3f}m, MAE={mae:.3f}m, R²={r2:.3f}")
    
    return predictions, metrics, depths_sample, features_sample

def create_comparison_plots(predictions, metrics, depths_true, output_dir):
    """Create comprehensive comparison visualizations"""
    
    print("\n📊 Creating Visualization Plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Palk Strait - Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Metrics comparison
    model_names = list(metrics.keys())
    rmse_values = [metrics[m]['RMSE'] for m in model_names]
    mae_values = [metrics[m]['MAE'] for m in model_names]
    r2_values = [metrics[m]['R²'] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    
    ax1.bar(x_pos, rmse_values, alpha=0.8, color='coral')
    ax1.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
    ax1.set_ylabel('RMSE (meters)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x_pos, mae_values, alpha=0.8, color='lightblue')
    ax2.set_title('Mean Absolute Error (MAE)', fontweight='bold')
    ax2.set_ylabel('MAE (meters)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    ax3.bar(x_pos, r2_values, alpha=0.8, color='lightgreen')
    ax3.set_title('R-squared Score', fontweight='bold')
    ax3.set_ylabel('R² Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot: True vs Predicted (best model)
    best_model = min(metrics.keys(), key=lambda k: metrics[k]['RMSE'])
    best_predictions = predictions[best_model]
    
    ax4.scatter(depths_true, best_predictions, alpha=0.6, s=20)
    ax4.plot([depths_true.min(), depths_true.max()], 
             [depths_true.min(), depths_true.max()], 'r--', linewidth=2)
    ax4.set_xlabel('True Depth (m)')
    ax4.set_ylabel('Predicted Depth (m)')
    ax4.set_title(f'True vs Predicted - {best_model}', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'palk_strait_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: palk_strait_model_comparison.png")
    
    # 2. Individual Model Performance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Palk Strait - Individual Model Analysis', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    for i, (name, pred) in enumerate(predictions.items()):
        ax = axes[i]
        
        # Scatter plot with residuals coloring
        residuals = depths_true - pred
        scatter = ax.scatter(depths_true, pred, c=residuals, cmap='RdYlBu', 
                           alpha=0.6, s=20, vmin=-2, vmax=2)
        
        ax.plot([depths_true.min(), depths_true.max()], 
                [depths_true.min(), depths_true.max()], 'k--', linewidth=1)
        
        ax.set_xlabel('True Depth (m)')
        ax.set_ylabel('Predicted Depth (m)')
        ax.set_title(f'{name}\nRMSE: {metrics[name]["RMSE"]:.3f}m')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Residual (m)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'palk_strait_individual_models.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: palk_strait_individual_models.png")
    
    # 3. Error Distribution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Palk Strait - Error Distribution Analysis', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    for i, (name, pred) in enumerate(predictions.items()):
        ax = axes[i]
        
        residuals = depths_true - pred
        
        # Histogram of residuals
        ax.hist(residuals, bins=30, alpha=0.7, density=True, color=f'C{i}')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (m)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} - Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        ax.text(0.05, 0.95, f'Mean: {residuals.mean():.3f}m\nStd: {residuals.std():.3f}m', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'palk_strait_error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: palk_strait_error_distribution.png")
    
    # 4. Save metrics to CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(output_dir / 'palk_strait_model_metrics.csv')
    print(f"   Saved: palk_strait_model_metrics.csv")
    
    return best_model

def create_summary_report(metrics, best_model, output_dir):
    """Create a summary report"""
    
    print("\n📋 Creating Summary Report...")
    
    report = f"""
# Palk Strait - Satellite Bathymetry Analysis Report

## Region Overview
- **Location**: Palk Strait (9.0°N - 10.5°N, 78.5°E - 80.0°E)
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Model Performance Summary

| Model | RMSE (m) | MAE (m) | R² Score |
|-------|----------|---------|----------|
"""
    
    for name, metric in metrics.items():
        report += f"| {name} | {metric['RMSE']:.3f} | {metric['MAE']:.3f} | {metric['R²']:.3f} |\n"
    
    report += f"""
## Best Performing Model
**{best_model}** achieved the lowest RMSE of {metrics[best_model]['RMSE']:.3f}m

## Generated Visualizations
1. `palk_strait_model_comparison.png` - Overall model performance comparison
2. `palk_strait_individual_models.png` - Individual model scatter plots with residuals
3. `palk_strait_error_distribution.png` - Error distribution histograms
4. `palk_strait_model_metrics.csv` - Detailed metrics in CSV format

## Recommendations
- Use **{best_model}** for operational bathymetry mapping in this region
- Consider ensemble methods for improved accuracy
- Validate results with in-situ measurements when available
"""
    
    with open(output_dir / 'palk_strait_analysis_report.md', 'w') as f:
        f.write(report)
    
    print(f"   Saved: palk_strait_analysis_report.md")

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("PALK STRAIT BATHYMETRY VISUALIZATION GENERATOR")
    print("=" * 80)
    
    try:
        # Load data and models
        features, depths, scaler, models, metadata, output_dir = load_palk_strait_data()
        
        # Generate predictions
        predictions, metrics, depths_sample, features_sample = generate_model_predictions(
            features, depths, scaler, models
        )
        
        # Create visualizations
        best_model = create_comparison_plots(predictions, metrics, depths_sample, output_dir)
        
        # Create summary report
        create_summary_report(metrics, best_model, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS: All visualizations generated successfully!")
        print(f"📁 Output directory: {output_dir}")
        print("🌊 Palk Strait bathymetry analysis complete!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)