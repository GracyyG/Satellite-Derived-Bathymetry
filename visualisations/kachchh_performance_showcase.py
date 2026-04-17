#!/usr/bin/env python3
"""
Simple Model Performance Visualization for Kachchh Region
Showcases the improved RMSE performance with realistic bathymetry data
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_test_data_and_models(region_name):
    """Load test data and trained models"""
    
    project_root = Path(__file__).parent.parent
    
    # Load test data
    test_data_dir = project_root / f"data/processed/{region_name}/training_data/test"
    X_test = np.load(test_data_dir / "features.npy")
    y_test = np.load(test_data_dir / "depths.npy")
    
    # Remove depth_value feature (index 8) to match spectral training
    X_test_spectral = np.delete(X_test, 8, axis=1)
    
    print(f"[OK] Test data loaded: {X_test_spectral.shape[0]} samples")
    print(f"    Depth range: {y_test.min():.1f}m to {y_test.max():.1f}m")
    
    # Load models
    models_dir = project_root / f"models/{region_name}"
    models = {}
    
    model_files = {
        'Linear Regression': 'linear_regression.joblib',
        'Random Forest': 'random_forest.joblib', 
        'XGBoost': 'xgboost.joblib'
    }
    
    for name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            models[name] = joblib.load(model_path)
            print(f"[OK] {name} model loaded")
    
    # Load metrics
    metrics_file = models_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
    else:
        saved_metrics = {}
    
    return X_test_spectral, y_test, models, saved_metrics

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def create_performance_visualizations(models, X_test, y_test, region_name, output_dir):
    """Create comprehensive performance visualizations"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Generate predictions and calculate metrics
    results = {}
    predictions = {}
    
    print("\nGenerating predictions and calculating metrics...")
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            metrics = calculate_metrics(y_test, y_pred)
            results[name] = metrics
            
            print(f"{name:20} | RMSE: {metrics['rmse']:6.3f}m | MAE: {metrics['mae']:6.3f}m | R²: {metrics['r2']:6.3f}")
        except Exception as e:
            print(f"[ERROR] {name} prediction failed: {e}")
    
    if not results:
        print("[ERROR] No successful predictions generated")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. True vs Predicted scatter plots
    n_models = len(predictions)
    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = plt.subplot(3, n_models, i + 1)
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=15)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Formatting
        ax.set_xlabel('True Depth (m)')
        ax.set_ylabel('Predicted Depth (m)')
        ax.set_title(f'{name}\nRMSE: {results[name]["rmse"]:.3f}m')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 2. Residual plots
    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = plt.subplot(3, n_models, i + 1 + n_models)
        
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=15)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        # Add residual statistics
        residual_std = np.std(residuals)
        ax.axhline(y=2*residual_std, color='orange', linestyle=':', alpha=0.7, label='±2σ')
        ax.axhline(y=-2*residual_std, color='orange', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Predicted Depth (m)')
        ax.set_ylabel('Residuals (m)')
        ax.set_title(f'{name} - Residuals\nStd: {residual_std:.3f}m')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 3. Performance comparison bar chart
    ax = plt.subplot(3, 1, 3)
    
    model_names = list(results.keys())
    rmse_values = [results[name]['rmse'] for name in model_names]
    mae_values = [results[name]['mae'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
    bars2 = ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars1, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}m', ha='center', va='bottom', fontweight='bold')
    
    for bar, value in zip(bars2, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}m', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Error (meters)')
    ax.set_title(f'Model Performance Comparison - {region_name.title()} Region\nRealistic Bathymetry Training Data')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / f"{region_name}_realistic_bathymetry_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Performance visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_csv = output_dir / f"{region_name}_model_performance_metrics.csv"
    results_df.to_csv(results_csv)
    print(f"[OK] Detailed metrics saved to: {results_csv}")
    
    return results

def main():
    region_name = "kachchh"
    
    print("="*80)
    print(f"REALISTIC BATHYMETRY MODEL PERFORMANCE - {region_name.upper()} REGION")
    print("="*80)
    
    # Load data and models
    X_test, y_test, models, saved_metrics = load_test_data_and_models(region_name)
    
    if not models:
        print("[ERROR] No models found!")
        return
    
    # Create output directory
    output_dir = Path(f"outputs/{region_name}/realistic_bathymetry_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    results = create_performance_visualizations(models, X_test, y_test, region_name, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    
    print(f"Region: {region_name.title()}")
    print(f"Test samples: {len(y_test)}")
    print(f"Depth range: {y_test.min():.1f}m - {y_test.max():.1f}m")
    print(f"Mean depth: {y_test.mean():.1f}m ± {y_test.std():.1f}m")
    print(f"\nBest model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.3f}m)")
    
    print(f"\nResults saved to: {output_dir}")
    print("\n🎉 Realistic bathymetry performance analysis complete!")

if __name__ == "__main__":
    main()