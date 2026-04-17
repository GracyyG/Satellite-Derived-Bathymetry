"""
Satellite-Derived Bathymetry Model Module

This module implements machine learning models for SDB prediction using
preprocessed Sentinel-2 data and GEBCO reference points.

Classes:
    - SDBModel: Base class for SDB models
    - RandomForestSDB: Random Forest regression model for SDB
    - XGBoostSDB: XGBoost regression model for SDB
    - SVRSDB: Support Vector Regression model for SDB
"""

from typing import Dict, Union, Tuple, List
import numpy as np
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import rasterio
from rasterio.transform import from_origin
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDBModel:
    """Base class for Satellite-Derived Bathymetry models."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Train the SDB model.
        
        Args:
            X: Feature array
            y: Target depths
            feature_names: List of feature names for importance ranking
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (metrics dict, prediction arrays dict)
        """
        try:
            # Store feature names
            self.feature_names = feature_names
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='neg_root_mean_squared_error'
            )
            
            # Calculate metrics
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'cv_rmse_mean': -cv_scores.mean(),
                'cv_rmse_std': cv_scores.std()
            }
            
            # Store predictions
            predictions = {
                'y_train': y_train,
                'y_test': y_test,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
            
            logger.info(f"Training completed. Test RMSE: {metrics['test_rmse']:.2f}m")
            return metrics, predictions
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make depth predictions.
        
        Args:
            X: Feature array
            
        Returns:
            Array of predicted depths
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_raster(
        self,
        features_ds: xr.Dataset,
        water_mask: np.ndarray,
        output_path: Path = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict bathymetry for entire raster image.
        
        Args:
            features_ds: xarray Dataset containing feature bands
            water_mask: Boolean array marking water pixels
            output_path: Optional path to save GeoTIFF output
            
        Returns:
            Tuple of (bathymetry array, raster metadata)
        """
        try:
            # Extract feature array for water pixels
            feature_arrays = []
            for feature in self.feature_names:
                feature_arrays.append(features_ds[feature].values[water_mask])
            
            X = np.column_stack(feature_arrays)
            
            # Make predictions
            depth_predictions = self.predict(X)
            
            # Create full raster
            bathymetry = np.full_like(water_mask, np.nan, dtype=np.float32)
            bathymetry[water_mask] = depth_predictions
            
            if output_path is not None:
                # Get spatial reference info from input
                transform = features_ds.rio.transform()
                crs = features_ds.rio.crs
                
                # Save as GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=bathymetry.shape[0],
                    width=bathymetry.shape[1],
                    count=1,
                    dtype=np.float32,
                    crs=crs,
                    transform=transform,
                    nodata=np.nan
                ) as dst:
                    dst.write(bathymetry, 1)
                logger.info(f"Saved bathymetry predictions to {output_path}")
            
            return bathymetry, {
                'transform': transform,
                'crs': crs
            }
            
        except Exception as e:
            logger.error(f"Raster prediction failed: {str(e)}")
            raise
    
    def save_model(self, path: Path) -> None:
        """Save model and scaler to file."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        
    def load_model(self, path: Path) -> None:
        """Load model and scaler from file."""
        if not path.exists():
            raise ValueError(f"Model file not found: {path}")
            
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_names = saved_data['feature_names']
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores if available."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        if self.feature_names is None:
            return dict(enumerate(self.model.feature_importances_))
            
        return dict(zip(self.feature_names, self.model.feature_importances_))

class RandomForestSDB(SDBModel):
    """Random Forest implementation of SDB model."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt'
    ):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=42
        )

class XGBoostSDB(SDBModel):
    """XGBoost implementation of SDB model."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8
    ):
        super().__init__()
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            n_jobs=-1,
            random_state=42
        )

class SVRSDB(SDBModel):
    """Support Vector Regression implementation of SDB model."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        epsilon: float = 0.1
    ):
        super().__init__()
        self.model = SVR(
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon
        )
        
if __name__ == "__main__":
    try:
        # Load preprocessed features and depths
        data_dir = Path('data/processed')
        features = np.load(data_dir / 'features.npy')
        depths = np.load(data_dir / 'depths.npy')
        
        # Load preprocessed dataset for raster prediction
        features_ds = xr.open_dataset(data_dir / 'processed_data.nc')
        water_mask = features_ds['water_mask'].values.astype(bool)
        
        # Define feature names
        feature_names = [
            'B02', 'B03', 'B04', 'B08',  # Raw bands
            'NDWI', 'MNDWI', 'SR', 'BR_ratio',  # Indices
            'B02/B03', 'B02/B04', 'B03/B04'  # Band ratios
        ]
        
        # Initialize models
        models = {
            'RandomForest': RandomForestSDB(n_estimators=200),
            'XGBoost': XGBoostSDB(n_estimators=200),
            'SVR': SVRSDB(kernel='rbf')
        }
        
        # Dictionary to store model performance
        model_performance = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            # Train model
            metrics, predictions = model.train(
                features, depths,
                feature_names=feature_names
            )
            
            # Store performance metrics
            model_performance[name] = metrics['test_rmse']
            
            # Print metrics
            logger.info(f"{name} Results:")
            logger.info(f"Train RMSE: {metrics['train_rmse']:.2f}m")
            logger.info(f"Test RMSE: {metrics['test_rmse']:.2f}m")
            logger.info(f"Train R²: {metrics['train_r2']:.3f}")
            logger.info(f"Test R²: {metrics['test_r2']:.3f}")
            
            if name in ['RandomForest', 'XGBoost']:
                # Get feature importance
                importance = model.get_feature_importance()
                sorted_features = sorted(
                    importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                logger.info("\nFeature Importance:")
                for feature, score in sorted_features[:5]:
                    logger.info(f"{feature}: {score:.3f}")
            
            # Save model
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)
            model.save_model(model_dir / f'{name.lower()}_model.joblib')
        
        # Select best model based on test RMSE
        best_model_name = min(model_performance.items(), key=lambda x: x[1])[0]
        best_model = models[best_model_name]
        
        logger.info(f"\nBest model: {best_model_name}")
        
        # Predict bathymetry for entire raster
        logger.info("Predicting bathymetry for entire area...")
        bathymetry, metadata = best_model.predict_raster(
            features_ds,
            water_mask,
            output_path=data_dir / 'predicted_bathymetry.tif'
        )
        
        logger.info("\nModel training, evaluation, and prediction completed successfully!")
        logger.info(f"Final bathymetry raster shape: {bathymetry.shape}")
        
    except Exception as e:
        logger.error(f"Error during model testing: {str(e)}")