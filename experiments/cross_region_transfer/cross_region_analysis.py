#!/usr/bin/env python3
"""
Cross-Region Transfer Learning Experiment
Testing Kachchh-trained models on Goa region data

This experiment evaluates how well models trained on Kachchh coastal waters
perform when applied to Goa region bathymetry data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

class CrossRegionTransferAnalysis:
    """Cross-region transfer learning analysis"""
    
    def __init__(self, source_region="kachchh", target_region="goa"):
        self.source_region = source_region
        self.target_region = target_region
        self.project_root = Path(__file__).parent.parent.parent
        self.output_dir = Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("CROSS-REGION TRANSFER LEARNING EXPERIMENT")
        print("="*80)
        print(f"Source Region (Trained): {source_region.title()}")
        print(f"Target Region (Testing): {target_region.title()}")
        print(f"Output Directory: {self.output_dir}")
    
    def load_source_models(self):
        """Load trained models from source region (Kachchh)"""
        
        models_dir = self.project_root / f"models/{self.source_region}"
        
        print(f"\n[LOADING] Source models from {models_dir}")
        
        self.models = {}
        model_files = {
            'XGBoost': 'xgboost.joblib',
            'Random Forest': 'random_forest.joblib',
            'Linear Regression': 'linear_regression.joblib'
        }
        
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                print(f"  ✅ {name} loaded from {self.source_region}")
            else:
                print(f"  ❌ {name} not found: {model_path}")
        
        # Load source scaler
        scaler_path = models_dir / "feature_scaler.joblib"
        if scaler_path.exists():
            self.source_scaler = joblib.load(scaler_path)
            print(f"  ✅ Feature scaler loaded from {self.source_region}")
        else:
            print(f"  ❌ Feature scaler not found")
            self.source_scaler = None
        
        # Load source metrics for comparison
        metrics_path = models_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.source_metrics = json.load(f)
            print(f"  ✅ Source performance metrics loaded")
        else:
            self.source_metrics = {}
        
        return len(self.models) > 0
    
    def load_target_data(self):
        """Load target region data (Goa)"""
        
        print(f"\n[LOADING] Target region data from {self.target_region}")
        
        # First try processed training data
        target_data_dir = self.project_root / f"data/processed/{self.target_region}/training_data"
        
        if not target_data_dir.exists():
            # Try sentinel processed data
            target_data_dir = self.project_root / f"data/sentinel/{self.target_region}/processed/training_data"
        
        if not target_data_dir.exists():
            print(f"  ❌ No training data found for {self.target_region}")
            return False
        
        # Load features and depths
        features_file = target_data_dir / "features.npy"
        depths_file = target_data_dir / "depths.npy"
        
        if features_file.exists() and depths_file.exists():
            self.target_features = np.load(features_file)
            self.target_depths = np.load(depths_file)
            
            print(f"  ✅ Features loaded: {self.target_features.shape}")
            print(f"  ✅ Depths loaded: {self.target_depths.shape}")
            
            # Load coordinates if available
            coords_file = target_data_dir / "coordinates.npy"
            if coords_file.exists():
                self.target_coords = np.load(coords_file)
                print(f"  ✅ Coordinates loaded: {self.target_coords.shape}")
            else:
                # Generate synthetic coordinates
                n_samples = len(self.target_features)
                self.target_coords = np.random.uniform(
                    [15.0, 73.0], [15.5, 74.0], (n_samples, 2)
                )
                print(f"  ⚠️  Generated synthetic coordinates for visualization")
            
            return True
        else:
            print(f"  ❌ Required data files not found in {target_data_dir}")
            return False
    
    def prepare_target_features(self):
        """Prepare target features to match source model expectations"""
        
        print(f"\n[PREPROCESSING] Preparing target features for model compatibility")
        
        # Handle different feature dimensions
        source_n_features = 12  # Kachchh spectral features (without depth_value)
        target_features = self.target_features.copy()
        
        print(f"  Target features shape: {target_features.shape}")
        
        # If target has depth_value feature, remove it
        if target_features.shape[1] == 13:
            target_features = np.delete(target_features, 8, axis=1)  # Remove depth_value
            print(f"  ✅ Removed depth_value feature: {target_features.shape}")
        
        # Handle dimension mismatch
        if target_features.shape[1] < source_n_features:
            # Pad with zeros
            padding = np.zeros((target_features.shape[0], source_n_features - target_features.shape[1]))
            target_features = np.concatenate([target_features, padding], axis=1)
            print(f"  ✅ Padded features to match source: {target_features.shape}")
        elif target_features.shape[1] > source_n_features:
            # Truncate
            target_features = target_features[:, :source_n_features]
            print(f"  ✅ Truncated features to match source: {target_features.shape}")
        
        # Scale features using source scaler if available
        if self.source_scaler is not None:
            target_features = self.source_scaler.transform(target_features)
            print(f"  ✅ Applied source region feature scaling")
        
        self.target_features_processed = target_features
        
        # Clean data
        valid_mask = ~np.isnan(target_features).any(axis=1) & ~np.isnan(self.target_depths)
        self.target_features_processed = target_features[valid_mask]
        self.target_depths_clean = self.target_depths[valid_mask]
        self.target_coords_clean = self.target_coords[valid_mask]
        
        print(f"  ✅ Clean samples: {len(self.target_features_processed)}")
        print(f"  📊 Target depth range: {self.target_depths_clean.min():.1f}m to {self.target_depths_clean.max():.1f}m")
        
        return True
    
    def run_transfer_predictions(self):
        """Run predictions using source models on target data"""
        
        print(f"\n[TRANSFER] Running cross-region predictions")
        
        self.transfer_results = {}
        self.predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(self.target_features_processed)
                self.predictions[model_name] = y_pred
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(self.target_depths_clean, y_pred))
                mae = mean_absolute_error(self.target_depths_clean, y_pred)
                r2 = r2_score(self.target_depths_clean, y_pred)
                
                self.transfer_results[model_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'source_rmse': self.source_metrics.get(f'{model_name.lower().replace(" ", "_")}_spectral', {}).get('rmse', 0),
                }
                
                print(f"  {model_name:20} | RMSE: {rmse:7.3f}m | MAE: {mae:6.3f}m | R²: {r2:7.3f}")
                
            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")
        
        return len(self.transfer_results) > 0
    
    def create_transfer_analysis_plots(self):
        """Create comprehensive transfer learning analysis plots"""
        
        print(f"\n[VISUALIZATION] Creating transfer learning analysis plots")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Cross-Region Transfer Learning Analysis\\n' +
                     f'{self.source_region.title()} → {self.target_region.title()}', fontsize=16)
        
        models = list(self.transfer_results.keys())
        
        # 1. Performance comparison (Source vs Transfer)
        ax = axes[0, 0]
        source_rmse = [self.transfer_results[m]['source_rmse'] for m in models]
        transfer_rmse = [self.transfer_results[m]['rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, source_rmse, width, label=f'{self.source_region.title()} (Source)', alpha=0.8)
        bars2 = ax.bar(x + width/2, transfer_rmse, width, label=f'{self.target_region.title()} (Transfer)', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('RMSE (m)')
        ax.set_title('Source vs Transfer Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\\n') for m in models])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, source_rmse):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}m', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, transfer_rmse):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}m', ha='center', va='bottom', fontsize=9)
        
        # 2. True vs Predicted (Best model)
        best_model = min(self.transfer_results.items(), key=lambda x: x[1]['rmse'])[0]
        ax = axes[0, 1]
        
        y_true = self.target_depths_clean
        y_pred = self.predictions[best_model]
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=15)
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('True Depth (m)')
        ax.set_ylabel('Predicted Depth (m)')
        ax.set_title(f'{best_model} - True vs Predicted\\nRMSE: {self.transfer_results[best_model]["rmse"]:.3f}m')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Residuals analysis
        ax = axes[0, 2]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=15)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Depth (m)')
        ax.set_ylabel('Residuals (m)')
        ax.set_title(f'{best_model} - Residuals Analysis')
        ax.grid(True, alpha=0.3)
        
        # 4. Geographic distribution (if coordinates available)
        ax = axes[1, 0]
        if hasattr(self, 'target_coords_clean'):
            lats = self.target_coords_clean[:, 0]
            lons = self.target_coords_clean[:, 1]
            scatter = ax.scatter(lons, lats, c=y_true, cmap='Blues_r', s=15, alpha=0.7)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('True Bathymetry Distribution')
            plt.colorbar(scatter, ax=ax, label='Depth (m)')
        else:
            ax.text(0.5, 0.5, 'No coordinate data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Geographic Distribution')
        
        # 5. Prediction distribution
        ax = axes[1, 1]
        if hasattr(self, 'target_coords_clean'):
            scatter = ax.scatter(lons, lats, c=y_pred, cmap='viridis', s=15, alpha=0.7)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'{best_model} Predictions')
            plt.colorbar(scatter, ax=ax, label='Predicted Depth (m)')
        else:
            ax.text(0.5, 0.5, 'No coordinate data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Prediction Distribution')
        
        # 6. Error distribution
        ax = axes[1, 2]
        errors = np.abs(residuals)
        ax.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(errors.mean(), color='red', linestyle='--', lw=2,
                  label=f'Mean Error: {errors.mean():.3f}m')
        ax.set_xlabel('Absolute Error (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Depth stratified performance
        ax = axes[2, 0]
        depth_ranges = [(0, 10), (10, 25), (25, 50), (50, 100)]
        range_names = ['0-10m', '10-25m', '25-50m', '50-100m']
        range_rmse = []
        
        for min_d, max_d in depth_ranges:
            mask = (y_true >= min_d) & (y_true < max_d)
            if np.sum(mask) > 5:
                range_rmse.append(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            else:
                range_rmse.append(0)
        
        bars = ax.bar(range_names, range_rmse, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
        ax.set_xlabel('Depth Range')
        ax.set_ylabel('RMSE (m)')
        ax.set_title('Performance by Depth Range')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, range_rmse):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}m', ha='center', va='bottom', fontsize=10)
        
        # 8. Model comparison metrics
        ax = axes[2, 1]
        ax.axis('off')
        
        # Create comparison table
        table_data = []
        for model_name, metrics in self.transfer_results.items():
            source_rmse = metrics['source_rmse']
            transfer_rmse = metrics['rmse']
            degradation = ((transfer_rmse - source_rmse) / source_rmse * 100) if source_rmse > 0 else 0
            
            table_data.append([
                model_name,
                f"{source_rmse:.3f}m",
                f"{transfer_rmse:.3f}m", 
                f"{degradation:+.1f}%"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', f'{self.source_region.title()} RMSE', f'{self.target_region.title()} RMSE', 'Change'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Transfer Learning Performance Summary')
        
        # 9. Transfer learning insights
        ax = axes[2, 2]
        ax.axis('off')
        
        # Calculate transfer insights
        best_transfer_rmse = min(self.transfer_results.values(), key=lambda x: x['rmse'])['rmse']
        best_source_rmse = min([m['source_rmse'] for m in self.transfer_results.values() if m['source_rmse'] > 0])
        
        insights_text = f"""
        TRANSFER LEARNING INSIGHTS
        
        Best Source RMSE: {best_source_rmse:.3f}m
        Best Transfer RMSE: {best_transfer_rmse:.3f}m
        
        Performance Change: {((best_transfer_rmse - best_source_rmse) / best_source_rmse * 100):+.1f}%
        
        Target Region: {self.target_region.title()}
        Sample Count: {len(y_true)}
        Depth Range: {y_true.min():.1f} - {y_true.max():.1f}m
        
        Best Transfer Model: {best_model}
        """
        
        ax.text(0.1, 0.9, insights_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"transfer_{self.source_region}_to_{self.target_region}_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✅ Transfer analysis plot saved: {plot_path}")
        
        return plot_path
    
    def create_interactive_transfer_plot(self):
        """Create interactive Plotly transfer analysis"""
        
        print(f"\n[INTERACTIVE] Creating interactive transfer analysis")
        
        # Get best model
        best_model = min(self.transfer_results.items(), key=lambda x: x[1]['rmse'])[0]
        y_true = self.target_depths_clean
        y_pred = self.predictions[best_model]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'True vs Predicted ({best_model})',
                'Performance Comparison',
                'Geographic Distribution - True',
                'Geographic Distribution - Predicted'
            ]
        )
        
        # True vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred, mode='markers',
                marker=dict(size=6, opacity=0.6),
                name='Predictions',
                text=[f'True: {t:.1f}m<br>Pred: {p:.1f}m<br>Error: {abs(t-p):.2f}m' 
                      for t, p in zip(y_true, y_pred)]
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                      line=dict(color='red', dash='dash'), name='Perfect'),
            row=1, col=1
        )
        
        # Performance comparison
        models = list(self.transfer_results.keys())
        source_rmse = [self.transfer_results[m]['source_rmse'] for m in models]
        transfer_rmse = [self.transfer_results[m]['rmse'] for m in models]
        
        fig.add_trace(
            go.Bar(x=models, y=source_rmse, name=f'{self.source_region.title()} (Source)',
                  text=[f'{v:.3f}m' for v in source_rmse], textposition='auto'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=models, y=transfer_rmse, name=f'{self.target_region.title()} (Transfer)',
                  text=[f'{v:.3f}m' for v in transfer_rmse], textposition='auto'),
            row=1, col=2
        )
        
        # Geographic plots (if coordinates available)
        if hasattr(self, 'target_coords_clean'):
            lats = self.target_coords_clean[:, 0]
            lons = self.target_coords_clean[:, 1]
            
            # True bathymetry
            fig.add_trace(
                go.Scatter(x=lons, y=lats, mode='markers',
                          marker=dict(size=6, color=y_true, colorscale='Blues_r'),
                          name='True Depth'),
                row=2, col=1
            )
            
            # Predicted bathymetry
            fig.add_trace(
                go.Scatter(x=lons, y=lats, mode='markers',
                          marker=dict(size=6, color=y_pred, colorscale='Viridis'),
                          name='Predicted Depth'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Transfer Learning Analysis: {self.source_region.title()} → {self.target_region.title()}<br>' +
                  f'Best Model: {best_model} (RMSE: {self.transfer_results[best_model]["rmse"]:.3f}m)',
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text="True Depth (m)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Depth (m)", row=1, col=1)
        
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_yaxes(title_text="RMSE (m)", row=1, col=2)
        
        if hasattr(self, 'target_coords_clean'):
            fig.update_xaxes(title_text="Longitude", row=2, col=1)
            fig.update_yaxes(title_text="Latitude", row=2, col=1)
            fig.update_xaxes(title_text="Longitude", row=2, col=2)
            fig.update_yaxes(title_text="Latitude", row=2, col=2)
        
        # Save interactive plot
        html_path = self.output_dir / f"transfer_{self.source_region}_to_{self.target_region}_interactive.html"
        fig.write_html(str(html_path))
        
        print(f"  ✅ Interactive transfer analysis saved: {html_path}")
        
        return html_path
    
    def save_transfer_results(self):
        """Save transfer learning results"""
        
        print(f"\n[SAVING] Transfer learning results")
        
        # Save metrics
        results_df = pd.DataFrame(self.transfer_results).T
        csv_path = self.output_dir / f"transfer_{self.source_region}_to_{self.target_region}_metrics.csv"
        results_df.to_csv(csv_path)
        
        # Save detailed results
        detailed_results = {
            'experiment': {
                'source_region': self.source_region,
                'target_region': self.target_region,
                'n_samples': len(self.target_depths_clean),
                'target_depth_range': [float(self.target_depths_clean.min()), float(self.target_depths_clean.max())],
                'target_depth_mean': float(self.target_depths_clean.mean())
            },
            'transfer_performance': self.transfer_results,
            'source_performance': self.source_metrics
        }
        
        json_path = self.output_dir / f"transfer_{self.source_region}_to_{self.target_region}_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"  ✅ Metrics saved: {csv_path}")
        print(f"  ✅ Detailed results saved: {json_path}")
        
        return csv_path, json_path
    
    def run_complete_analysis(self):
        """Run complete cross-region transfer analysis"""
        
        # Load source models
        if not self.load_source_models():
            print("❌ Failed to load source models")
            return False
        
        # Load target data
        if not self.load_target_data():
            print("❌ Failed to load target data")
            return False
        
        # Prepare target features
        if not self.prepare_target_features():
            print("❌ Failed to prepare target features")
            return False
        
        # Run transfer predictions
        if not self.run_transfer_predictions():
            print("❌ Failed to run transfer predictions")
            return False
        
        # Create visualizations
        plot_path = self.create_transfer_analysis_plots()
        html_path = self.create_interactive_transfer_plot()
        
        # Save results
        csv_path, json_path = self.save_transfer_results()
        
        # Print summary
        print(f"\n" + "="*80)
        print("CROSS-REGION TRANSFER ANALYSIS COMPLETE")
        print("="*80)
        
        best_model = min(self.transfer_results.items(), key=lambda x: x[1]['rmse'])
        print(f"✅ Best Transfer Model: {best_model[0]}")
        print(f"   Transfer RMSE: {best_model[1]['rmse']:.3f}m")
        print(f"   Source RMSE: {best_model[1]['source_rmse']:.3f}m")
        
        change_pct = ((best_model[1]['rmse'] - best_model[1]['source_rmse']) / best_model[1]['source_rmse'] * 100) if best_model[1]['source_rmse'] > 0 else 0
        print(f"   Performance Change: {change_pct:+.1f}%")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   • Static plot: {plot_path.name}")
        print(f"   • Interactive plot: {html_path.name}")
        print(f"   • Metrics: {csv_path.name}")
        print(f"   • Detailed results: {json_path.name}")
        
        return True

def main():
    """Main execution function"""
    
    # Create transfer analysis
    transfer_analysis = CrossRegionTransferAnalysis(source_region="kachchh", target_region="goa")
    
    # Run complete analysis
    success = transfer_analysis.run_complete_analysis()
    
    if success:
        print(f"\n🎉 Cross-region transfer learning experiment completed successfully!")
        print(f"🔬 This analysis shows how well Kachchh-trained models perform on Goa data")
        print(f"📊 Results can help understand model generalization across coastal regions")
    else:
        print(f"\n❌ Transfer learning experiment failed")

if __name__ == "__main__":
    main()