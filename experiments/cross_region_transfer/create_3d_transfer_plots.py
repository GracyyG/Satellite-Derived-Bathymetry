#!/usr/bin/env python3
"""
3D Visualization of Cross-Region Transfer Learning Results
Kachchh → Goa Transfer Learning with Interactive 3D Plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import json

class CrossRegionTransfer3D:
    """Generate comprehensive 3D visualizations for cross-region transfer learning"""
    
    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.output_dir = Path(__file__).parent / "3d_plots"
        self.output_dir.mkdir(exist_ok=True)
        
        print("🌊 CROSS-REGION TRANSFER LEARNING 3D VISUALIZATION")
        print("="*60)
        print(f"📁 Input: {self.results_dir}")
        print(f"📁 Output: {self.output_dir}")
    
    def load_transfer_data(self):
        """Load transfer learning results and target data"""
        
        print("\n[LOADING] Transfer learning data...")
        
        # Load detailed results
        detailed_file = self.results_dir / "transfer_kachchh_to_goa_detailed.json"
        with open(detailed_file) as f:
            self.detailed_results = json.load(f)
        
        # Load Goa training data
        goa_data_dir = Path("d:/Project/sdb_project/data/processed/goa/training_data")
        
        self.features = np.load(goa_data_dir / "features.npy")
        self.depths = np.load(goa_data_dir / "depths.npy")
        self.coords = np.load(goa_data_dir / "coordinates.npy")
        
        # Load Kachchh models for predictions
        models_dir = Path("d:/Project/sdb_project/models/kachchh")
        
        import joblib
        self.xgb_model = joblib.load(models_dir / "xgboost.joblib")
        self.rf_model = joblib.load(models_dir / "random_forest.joblib")
        self.scaler = joblib.load(models_dir / "feature_scaler.joblib")
        
        print(f"  ✅ Features: {self.features.shape}")
        print(f"  ✅ Target depths: {self.depths.shape}")
        print(f"  ✅ Coordinates: {self.coords.shape}")
        
        # Prepare features for prediction (remove depth_value feature)
        self.spectral_features = np.delete(self.features, 8, axis=1)  # Remove depth_value
        self.spectral_features_scaled = self.scaler.transform(self.spectral_features)
        
        # Generate predictions
        self.xgb_predictions = self.xgb_model.predict(self.spectral_features_scaled)
        self.rf_predictions = self.rf_model.predict(self.spectral_features_scaled)
        
        print(f"  ✅ Generated {len(self.xgb_predictions)} XGBoost predictions")
        print(f"  ✅ Generated {len(self.rf_predictions)} Random Forest predictions")
    
    def create_3d_bathymetry_surface(self):
        """Create interactive 3D bathymetry surfaces"""
        
        print("\n[3D SURFACE] Creating bathymetry surface plots...")
        
        # Sample data for better performance
        sample_size = 2000
        indices = np.random.choice(len(self.coords), sample_size, replace=False)
        
        lats = self.coords[indices, 0]
        lons = self.coords[indices, 1]
        true_depths = self.depths[indices]
        xgb_preds = self.xgb_predictions[indices]
        rf_preds = self.rf_predictions[indices]
        
        # Create subplots with 3D scenes
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}],
                   [{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=[
                'True Bathymetry (Goa Coastal Waters)',
                'XGBoost Predictions (Kachchh Model)',
                'Random Forest Predictions (Kachchh Model)', 
                'Prediction Errors (|True - Predicted|)'
            ],
            vertical_spacing=0.08
        )
        
        # True bathymetry surface
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=-true_depths,  # Negative for underwater
                mode='markers',
                marker=dict(
                    size=4,
                    color=true_depths,
                    colorscale='Blues_r',
                    colorbar=dict(title="Depth (m)", x=0.45),
                    opacity=0.8
                ),
                name='True Depth',
                hovertemplate='<b>True Bathymetry</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>Depth: %{marker.color:.1f}m<extra></extra>'
            ),
            row=1, col=1
        )
        
        # XGBoost predictions
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=-xgb_preds,
                mode='markers',
                marker=dict(
                    size=4,
                    color=xgb_preds,
                    colorscale='Viridis',
                    colorbar=dict(title="Predicted Depth (m)", x=1.02),
                    opacity=0.8
                ),
                name='XGBoost',
                hovertemplate='<b>XGBoost Prediction</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>Predicted: %{marker.color:.1f}m<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Random Forest predictions
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=-rf_preds,
                mode='markers',
                marker=dict(
                    size=4,
                    color=rf_preds,
                    colorscale='Plasma',
                    opacity=0.8
                ),
                name='Random Forest',
                hovertemplate='<b>Random Forest Prediction</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>Predicted: %{marker.color:.1f}m<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Prediction errors
        xgb_errors = np.abs(true_depths - xgb_preds)
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=-true_depths,
                mode='markers',
                marker=dict(
                    size=4,
                    color=xgb_errors,
                    colorscale='Reds',
                    colorbar=dict(title="Absolute Error (m)", x=1.02, y=0.3),
                    opacity=0.8
                ),
                name='Prediction Errors',
                hovertemplate='<b>XGBoost Error</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>True: %{customdata[0]:.1f}m<br>Pred: %{customdata[1]:.1f}m<br>Error: %{marker.color:.1f}m<extra></extra>',
                customdata=np.column_stack([true_depths, xgb_preds])
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Cross-Region Transfer Learning: 3D Bathymetry Analysis<br>' +
                     '<sub>Kachchh-trained models applied to Goa coastal waters</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            height=800,
            showlegend=False,
            font=dict(size=12)
        )
        
        # Update 3D scene properties
        scene_props = dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude", 
            zaxis_title="Depth (m, inverted)",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            aspectmode='cube'
        )
        
        fig.update_scenes(scene_props)
        
        # Save interactive plot
        html_path = self.output_dir / "transfer_3d_bathymetry_surfaces.html"
        fig.write_html(str(html_path))
        
        print(f"  ✅ 3D bathymetry surfaces saved: {html_path.name}")
        return html_path
    
    def create_3d_error_analysis(self):
        """Create 3D error distribution analysis"""
        
        print("\n[3D ERROR] Creating error distribution analysis...")
        
        # Calculate errors
        xgb_errors = np.abs(self.depths - self.xgb_predictions)
        rf_errors = np.abs(self.depths - self.rf_predictions)
        
        # Sample for performance
        sample_size = 3000
        indices = np.random.choice(len(self.coords), sample_size, replace=False)
        
        lats = self.coords[indices, 0]
        lons = self.coords[indices, 1]
        true_depths = self.depths[indices]
        xgb_errs = xgb_errors[indices]
        rf_errs = rf_errors[indices]
        
        # Create 3D error visualization
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=[
                'XGBoost Transfer Error Distribution',
                'Random Forest Transfer Error Distribution'
            ]
        )
        
        # XGBoost errors in 3D space
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=true_depths,
                mode='markers',
                marker=dict(
                    size=xgb_errs/5 + 2,  # Scale error to marker size
                    color=xgb_errs,
                    colorscale='Reds',
                    colorbar=dict(title="XGBoost Error (m)", x=0.45),
                    opacity=0.7,
                    line=dict(width=0.5, color='darkred')
                ),
                name='XGBoost Errors',
                hovertemplate='<b>XGBoost Transfer Error</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>Depth: %{z:.1f}m<br>Error: %{marker.color:.1f}m<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Random Forest errors in 3D space  
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=true_depths,
                mode='markers',
                marker=dict(
                    size=rf_errs/5 + 2,
                    color=rf_errs,
                    colorscale='Oranges',
                    colorbar=dict(title="Random Forest Error (m)", x=1.02),
                    opacity=0.7,
                    line=dict(width=0.5, color='darkorange')
                ),
                name='RF Errors',
                hovertemplate='<b>Random Forest Transfer Error</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>Depth: %{z:.1f}m<br>Error: %{marker.color:.1f}m<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='3D Error Distribution Analysis: Cross-Region Transfer Learning<br>' +
                     '<sub>Marker size = Error magnitude | Color intensity = Error value</sub>',
                x=0.5,
                font=dict(size=14)
            ),
            height=600,
            showlegend=False
        )
        
        # Update scenes
        fig.update_scenes(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="True Depth (m)",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.1))
        )
        
        html_path = self.output_dir / "transfer_3d_error_analysis.html"
        fig.write_html(str(html_path))
        
        print(f"  ✅ 3D error analysis saved: {html_path.name}")
        return html_path
    
    def create_3d_performance_comparison(self):
        """Create 3D performance comparison across depth ranges"""
        
        print("\n[3D PERFORMANCE] Creating performance comparison...")
        
        # Define depth bins
        depth_ranges = [(0, 5), (5, 15), (15, 25), (25, 40)]
        range_names = ['0-5m', '5-15m', '15-25m', '25-40m']
        
        # Calculate metrics for each depth range
        xgb_rmse_by_depth = []
        rf_rmse_by_depth = []
        sample_counts = []
        
        for min_d, max_d in depth_ranges:
            mask = (self.depths >= min_d) & (self.depths < max_d)
            if np.sum(mask) > 5:
                xgb_rmse = np.sqrt(np.mean((self.depths[mask] - self.xgb_predictions[mask])**2))
                rf_rmse = np.sqrt(np.mean((self.depths[mask] - self.rf_predictions[mask])**2))
                count = np.sum(mask)
            else:
                xgb_rmse = rf_rmse = count = 0
            
            xgb_rmse_by_depth.append(xgb_rmse)
            rf_rmse_by_depth.append(rf_rmse)
            sample_counts.append(count)
        
        # Create 3D performance comparison
        fig = go.Figure()
        
        # XGBoost performance bars (using scatter3d with size)
        fig.add_trace(
            go.Scatter3d(
                x=[0, 1, 2, 3],
                y=[0, 0, 0, 0],
                z=xgb_rmse_by_depth,
                mode='markers+text',
                marker=dict(
                    size=[max(20, rmse*2) for rmse in xgb_rmse_by_depth],
                    color='blue',
                    opacity=0.7,
                    symbol='square'
                ),
                text=[f'XGB<br>{rmse:.1f}m' for rmse in xgb_rmse_by_depth],
                textposition='middle center',
                name='XGBoost RMSE',
                hovertemplate='<b>XGBoost</b><br>Depth Range: %{customdata}<br>RMSE: %{z:.1f}m<br>Samples: %{text}<extra></extra>',
                customdata=range_names
            )
        )
        
        # Random Forest performance bars
        fig.add_trace(
            go.Scatter3d(
                x=[0.2, 1.2, 2.2, 3.2],
                y=[0, 0, 0, 0],
                z=rf_rmse_by_depth,
                mode='markers+text',
                marker=dict(
                    size=[max(20, rmse*2) for rmse in rf_rmse_by_depth],
                    color='orange',
                    opacity=0.7,
                    symbol='diamond'
                ),
                text=[f'RF<br>{rmse:.1f}m' for rmse in rf_rmse_by_depth],
                textposition='middle center',
                name='Random Forest RMSE',
                hovertemplate='<b>Random Forest</b><br>Depth Range: %{customdata}<br>RMSE: %{z:.1f}m<br>Samples: %{text}<extra></extra>',
                customdata=range_names
            )
        )
        
        fig.update_layout(
            title=dict(
                text='3D Performance Comparison by Depth Range<br>' +
                     '<sub>Cross-region transfer learning performance stratified by depth</sub>',
                x=0.5,
                font=dict(size=14)
            ),
            scene=dict(
                xaxis=dict(
                    title="Depth Range",
                    tickvals=[0.2, 1.2, 2.2, 3.2],
                    ticktext=range_names
                ),
                yaxis=dict(title="Model Comparison"),
                zaxis=dict(title="RMSE (m)"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=600
        )
        
        html_path = self.output_dir / "transfer_3d_performance_comparison.html"
        fig.write_html(str(html_path))
        
        print(f"  ✅ 3D performance comparison saved: {html_path.name}")
        return html_path
    
    def create_comprehensive_3d_showcase(self):
        """Create comprehensive 3D showcase with multiple views"""
        
        print("\n[3D SHOWCASE] Creating comprehensive 3D visualization...")
        
        # Sample data
        sample_size = 2500
        indices = np.random.choice(len(self.coords), sample_size, replace=False)
        
        lats = self.coords[indices, 0]
        lons = self.coords[indices, 1]
        true_depths = self.depths[indices]
        xgb_preds = self.xgb_predictions[indices]
        errors = np.abs(true_depths - xgb_preds)
        
        # Create comprehensive figure
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'scene', 'colspan': 2}, None, {'type': 'xy'}],
                [{'type': 'scene'}, {'type': 'scene'}, {'type': 'xy'}]
            ],
            subplot_titles=[
                'Interactive 3D Bathymetry Mesh (True vs Predicted)',
                'Transfer Learning Performance',
                'True vs Predicted Depth Scatter',
                'Error Distribution by Geographic Location',
                'Error Histogram'
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # Main 3D bathymetry comparison
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=-true_depths,
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.6),
                name='True Bathymetry',
                hovertemplate='True: %{z:.1f}m<br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=-xgb_preds,
                mode='markers', 
                marker=dict(size=3, color='red', opacity=0.6),
                name='XGBoost Predictions',
                hovertemplate='Predicted: %{z:.1f}m<br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # True vs Predicted scatter
        fig.add_trace(
            go.Scatter(
                x=true_depths, y=xgb_preds,
                mode='markers',
                marker=dict(size=4, color=errors, colorscale='Reds', opacity=0.6),
                name='Predictions',
                hovertemplate='True: %{x:.1f}m<br>Pred: %{y:.1f}m<br>Error: %{marker.color:.1f}m<extra></extra>'
            ),
            row=1, col=3
        )
        
        # Perfect prediction line
        max_depth = max(true_depths.max(), xgb_preds.max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_depth], y=[0, max_depth],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Perfect Prediction',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Geographic error distribution
        fig.add_trace(
            go.Scatter3d(
                x=lons, y=lats, z=errors,
                mode='markers',
                marker=dict(
                    size=5,
                    color=errors,
                    colorscale='Reds',
                    opacity=0.7
                ),
                name='Geographic Errors',
                hovertemplate='Error: %{z:.1f}m<br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Depth-stratified errors
        depth_bins = np.linspace(0, 40, 20)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
        bin_errors = []
        
        for i in range(len(depth_bins)-1):
            mask = (true_depths >= depth_bins[i]) & (true_depths < depth_bins[i+1])
            if np.sum(mask) > 0:
                bin_errors.append(errors[mask].mean())
            else:
                bin_errors.append(0)
        
        fig.add_trace(
            go.Scatter3d(
                x=bin_centers, y=[0]*len(bin_centers), z=bin_errors,
                mode='markers+lines',
                marker=dict(size=6, color='orange'),
                line=dict(color='orange', width=4),
                name='Error by Depth',
                hovertemplate='Depth: %{x:.1f}m<br>Mean Error: %{z:.2f}m<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Error histogram
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=30,
                name='Error Distribution',
                marker=dict(color='lightcoral', opacity=0.7)
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Comprehensive 3D Analysis: Kachchh→Goa Transfer Learning<br>' +
                     f'<sub>Best Transfer RMSE: {self.detailed_results["transfer_performance"]["XGBoost"]["rmse"]:.1f}m ' +
                     f'(vs Source: {self.detailed_results["transfer_performance"]["XGBoost"]["source_rmse"]:.1f}m)</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            height=900,
            showlegend=True
        )
        
        # Update scene properties
        fig.update_scenes(
            xaxis_title="Longitude",
            yaxis_title="Latitude", 
            zaxis_title="Depth/Error (m)",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.0))
        )
        
        # Update 2D subplot axes
        fig.update_xaxes(title_text="True Depth (m)", row=1, col=3)
        fig.update_yaxes(title_text="Predicted Depth (m)", row=1, col=3)
        fig.update_xaxes(title_text="Absolute Error (m)", row=2, col=3)
        fig.update_yaxes(title_text="Frequency", row=2, col=3)
        
        html_path = self.output_dir / "transfer_comprehensive_3d_showcase.html"
        fig.write_html(str(html_path))
        
        print(f"  ✅ Comprehensive 3D showcase saved: {html_path.name}")
        return html_path
    
    def generate_all_3d_plots(self):
        """Generate all 3D visualizations"""
        
        print("\n🚀 GENERATING ALL 3D VISUALIZATIONS...")
        
        # Load data
        self.load_transfer_data()
        
        # Generate all plots
        plots = []
        plots.append(self.create_3d_bathymetry_surface())
        plots.append(self.create_3d_error_analysis()) 
        plots.append(self.create_3d_performance_comparison())
        plots.append(self.create_comprehensive_3d_showcase())
        
        print(f"\n🎉 ALL 3D VISUALIZATIONS COMPLETE!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Generated {len(plots)} interactive 3D plots:")
        
        for i, plot_path in enumerate(plots, 1):
            print(f"   {i}. {plot_path.name}")
        
        return plots

def main():
    """Main execution function"""
    
    viz = CrossRegionTransfer3D()
    plots = viz.generate_all_3d_plots()
    
    print(f"\n🌊 Cross-region transfer learning 3D analysis complete!")
    print(f"💡 These plots show the dramatic performance degradation when applying")
    print(f"   Kachchh-trained models to Goa coastal waters (RMSE: ~1.2m → ~65m)")

if __name__ == "__main__":
    main()