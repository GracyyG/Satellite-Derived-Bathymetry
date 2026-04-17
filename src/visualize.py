"""
Visualization Module

This module provides functions for visualizing Satellite-Derived Bathymetry
results using matplotlib and plotly.

Functions:
    - plot_bathymetry_2d: Create 2D bathymetry plot
    - plot_bathymetry_3d: Create 3D surface plot of bathymetry
    - plot_error_distribution: Plot model error distribution
    - plot_comparison: Compare predicted vs reference depths
"""

from typing import Dict, Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import cmocean.cm as cmo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create custom colormap for bathymetry
bathymetry_colors = [
    '#000033',  # Deep water (dark blue)
    '#0000FF',  # Blue
    '#00FFFF',  # Cyan
    '#00FF00',  # Green
    '#FFFF00',  # Yellow
    '#FF8C00',  # Orange
    '#FF0000'   # Red (shallow water)
]
bathymetry_cmap = LinearSegmentedColormap.from_list('bathymetry', bathymetry_colors)

def plot_bathymetry_2d(
    depths: np.ndarray,
    coordinates: np.ndarray,
    output_path: Path,
    title: str = "Predicted Bathymetry",
    cmap: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create 2D bathymetry plot with color-coded depths.
    
    Args:
        depths: 2D array of depth values
        coordinates: Array of [longitude, latitude] coordinates
        output_path: Path to save the plot
        title: Plot title
        cmap: Optional colormap name (uses custom bathymetry cmap if None)
        figsize: Figure size in inches
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create plot
        im = plt.imshow(
            depths,
            cmap=cmap or bathymetry_cmap,
            extent=[
                coordinates[:, 0].min(),
                coordinates[:, 0].max(),
                coordinates[:, 1].min(),
                coordinates[:, 1].max()
            ]
        )
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Depth (m)', rotation=270, labelpad=15)
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"2D bathymetry plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"2D plotting failed: {str(e)}")
        raise

def plot_bathymetry_3d(
    depths: np.ndarray,
    coordinates: np.ndarray,
    output_path: Path,
    title: str = "3D Bathymetry Surface",
    colormap: Optional[str] = None,
    width: int = 1000,
    height: int = 800
) -> None:
    """
    Create interactive 3D surface plot of bathymetry using plotly.
    
    Args:
        depths: 2D array of depth values
        coordinates: Array of [longitude, latitude] coordinates
        output_path: Path to save the plot
        title: Plot title
        colormap: Optional plotly colormap name (e.g., 'Viridis', 'Blues')
        width: Plot width in pixels
        height: Plot height in pixels
    """
    try:
        # Create meshgrid for 3D surface
        lon_unique = np.unique(coordinates[:, 0])
        lat_unique = np.unique(coordinates[:, 1])
        lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(
                x=lon_mesh,
                y=lat_mesh,
                z=depths,
                colorscale=colormap or [
                    [0, '#000033'],    # Deep water (dark blue)
                    [0.2, '#0000FF'],  # Blue
                    [0.4, '#00FFFF'],  # Cyan
                    [0.6, '#00FF00'],  # Green
                    [0.8, '#FFFF00'],  # Yellow
                    [0.9, '#FF8C00'],  # Orange
                    [1.0, '#FF0000']   # Red (shallow water)
                ]
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Depth (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=width,
            height=height
        )
        
        # Save interactive HTML plot
        html_path = output_path.with_suffix('.html')
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path)
        
        # Also save static image for non-interactive viewing
        fig.write_image(output_path, width=width, height=height)
        
        logger.info(f"Interactive 3D plot saved to {html_path}")
        logger.info(f"Static 3D plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"3D plotting failed: {str(e)}")
        raise

def plot_error_distribution(
    predicted: np.ndarray,
    reference: np.ndarray,
    output_path: Path,
    title: str = "Depth Prediction Error Distribution",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50
) -> Dict[str, float]:
    """
    Plot model error distribution and return error metrics.
    
    Args:
        predicted: Predicted depth values
        reference: Reference depth values
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size in inches
        bins: Number of histogram bins
        
    Returns:
        Dictionary of error metrics including mean, std, rmse
    """
    try:
        # Calculate errors
        errors = predicted - reference
        
        # Calculate error metrics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot histogram
        plt.hist(errors, bins=bins, density=True, alpha=0.6, color='steelblue')
        
        # Add kernel density estimation
        kde = stats.gaussian_kde(errors)
        x_range = np.linspace(errors.min(), errors.max(), 200)
        plt.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        
        # Add vertical line at zero
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Add statistics annotations
        stats_text = (
            f'Mean Error: {mean_error:.2f}m\n'
            f'Std Dev: {std_error:.2f}m\n'
            f'RMSE: {rmse:.2f}m'
        )
        plt.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords='axes fraction',
            ha='right',
            va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8)
        )
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Error (m)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error distribution plot saved to {output_path}")
        
        # Return error metrics
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'rmse': rmse
        }
        
    except Exception as e:
        logger.error(f"Error distribution plotting failed: {str(e)}")
        raise

def plot_comparison(
    predicted: np.ndarray,
    reference: np.ndarray,
    output_path: Path,
    title: str = "Predicted vs Reference Depths",
    figsize: Tuple[int, int] = (10, 10),
    alpha: float = 0.5
) -> None:
    """
    Create scatter plot comparing predicted vs reference depths with
    regression line and statistics.
    
    Args:
        predicted: Predicted depth values
        reference: Reference depth values
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size in inches
        alpha: Transparency of scatter points
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        plt.scatter(reference, predicted, alpha=alpha, c='steelblue')
        
        # Add diagonal line (perfect prediction)
        min_val = min(predicted.min(), reference.min())
        max_val = max(predicted.max(), reference.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Calculate and plot regression line
        slope, intercept, r_value, _, _ = stats.linregress(reference, predicted)
        regression_line = slope * np.array([min_val, max_val]) + intercept
        plt.plot([min_val, max_val], regression_line, 'r-', alpha=0.8)
        
        # Add statistics annotations
        stats_text = (
            f'R² = {r_value**2:.3f}\n'
            f'Slope = {slope:.3f}\n'
            f'Intercept = {intercept:.2f}m'
        )
        plt.annotate(
            stats_text,
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            ha='left',
            va='top',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
        )
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Reference Depth (m)')
        plt.ylabel('Predicted Depth (m)')
        plt.grid(True, alpha=0.3)
        
        # Make plot square with equal axes
        plt.axis('square')
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Comparison plotting failed: {str(e)}")
        raise


# ============================================================================
# ADVANCED VISUALIZATION INTEGRATION SYSTEM
# ============================================================================

class AdvancedVisualizationManager:
    """Manages and executes advanced visualization functions from external modules"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.visualizations_dir = project_root / 'visualisations'
        self.viz_modules = {}
        self.viz_functions = {}
        
    def discover_visualization_modules(self) -> list:
        """Discover visualization modules in the visualisations directory"""
        viz_files = []
        
        if self.visualizations_dir.exists():
            for py_file in self.visualizations_dir.glob("*.py"):
                # Skip __init__.py and main_pipeline.py
                if py_file.name not in ['__init__.py', 'main_pipeline.py']:
                    viz_files.append(py_file.stem)
                    
        return viz_files
    
    def load_visualization_modules(self) -> dict:
        """Safely load visualization modules and extract functions"""
        import sys
        import importlib.util
        
        discovered_modules = self.discover_visualization_modules()
        
        print(f"🎨 Discovered {len(discovered_modules)} visualization modules:")
        for module_name in discovered_modules:
            print(f"  - {module_name}")
        
        loaded_modules = {}
        
        for module_name in discovered_modules:
            try:
                # Load module from file path
                module_path = self.visualizations_dir / f"{module_name}.py"
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                
                # Add to sys.modules to make imports work
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                loaded_modules[module_name] = module
                
                # Extract main function if it exists
                if hasattr(module, 'main'):
                    self.viz_functions[f"{module_name}.main"] = module.main
                
                print(f"✅ Loaded {module_name}")
                
            except Exception as e:
                print(f"⚠️  Could not load {module_name}: {e}")
                continue
        
        return loaded_modules
    
    def execute_visualizations(self, output_dir: Path) -> list:
        """Execute all discovered visualization functions"""
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        print(f"\n🎨 Executing {len(self.viz_functions)} advanced visualization functions...")
        
        for func_name, func in self.viz_functions.items():
            try:
                print(f"🔄 Running {func_name}...")
                
                # Call the visualization function
                # Most of your viz scripts are standalone, so we'll run them as modules
                result = self._execute_viz_script(func_name.split('.')[0])
                
                if result:
                    generated_files.extend(result)
                    print(f"✅ {func_name} completed successfully")
                else:
                    print(f"⚠️  {func_name} completed but returned no files")
                    
            except Exception as e:
                print(f"❌ Error in {func_name}: {e}")
                continue
        
        return generated_files
    
    def _execute_viz_script(self, module_name: str) -> list:
        """Execute a visualization script as a subprocess to avoid conflicts"""
        import subprocess
        import sys
        import os
        
        try:
            # Path to the visualization script
            script_path = self.visualizations_dir / f"{module_name}.py"
            
            if not script_path.exists():
                print(f"⚠️  Script not found: {script_path}")
                return []
            
            # Change to project directory for execution
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                # Execute the script
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    print(f"  ✅ {module_name} executed successfully")
                    # Look for generated files in the output directory
                    return self._find_generated_files(module_name)
                else:
                    print(f"  ❌ {module_name} failed with return code {result.returncode}")
                    if result.stderr:
                        print(f"  Error: {result.stderr[:200]}...")
                    return []
                    
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {module_name} timed out after 5 minutes")
            return []
        except Exception as e:
            print(f"  ❌ Error executing {module_name}: {e}")
            return []
    
    def _find_generated_files(self, module_name: str) -> list:
        """Find files generated by a visualization script"""
        generated_files = []
        
        # Common output patterns
        search_patterns = [
            f"**/*{module_name}*",
            "**/3d_*.html",
            "**/3d_*.png", 
            "**/*_comparison*.html",
            "**/*_comparison*.png",
            "**/*_heatmap*.png",
            "**/*_surface*.html",
            "**/*_stratified*.html"
        ]
        
        # Search in outputs directory
        for pattern in search_patterns:
            for file_path in self.project_root.glob(f"outputs/{pattern}"):
                if file_path.is_file():
                    # Get relative path for cleaner output
                    rel_path = file_path.relative_to(self.project_root)
                    generated_files.append(str(rel_path))
        
        return generated_files


def run_advanced_visualizations(output_dir: Path) -> list:
    """Main function to run all advanced visualizations"""
    
    # Initialize the visualization manager
    project_root = Path(__file__).parent.parent  # Go up from src/ to project root
    viz_manager = AdvancedVisualizationManager(project_root)
    
    # Load visualization modules
    viz_manager.load_visualization_modules()
    
    # Execute visualizations
    generated_files = viz_manager.execute_visualizations(output_dir)
    
    print("\n🎉 Advanced visualizations complete!")
    print(f"📁 Generated {len(generated_files)} files")
    
    return generated_files