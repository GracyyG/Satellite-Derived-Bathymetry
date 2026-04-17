# Satellite-Derived Bathymetry (SDB) Project

This project implements a workflow for deriving bathymetry from Sentinel-2 satellite imagery using machine learning approaches, validated against GEBCO reference data.

## Overview

### Key Features

- Automated Sentinel-2 data acquisition and preprocessing
- Multiple ML models for depth estimation (Random Forest, XGBoost, SVR)
- Interactive 2D and 3D bathymetry visualization
- Comprehensive error analysis and model comparison
- Configurable processing pipeline

### Applications

- Coastal bathymetry mapping
- Shallow water monitoring
- Coastal change detection
- Marine habitat mapping support
- Preliminary survey planning

## Project Structure

```
sdb_project/
├── data/
│   ├── sentinel/          # Sentinel-2 bands (B02, B03, B04, B08)
│   ├── gebco_reference/   # Bathymetry reference points
│   └── aoi.geojson       # Area of Interest (optional)
├── notebooks/
│   └── 01_sdb_processing.ipynb  # Main SDB workflow notebook
├── src/
│   ├── download_sentinel.py     # Sentinel-2 data download
│   ├── preprocess.py           # Image preprocessing
│   ├── sdb_model.py            # ML model implementation
│   ├── visualize.py           # Visualization tools
│   └── main.py                # Pipeline orchestration
├── output/
│   ├── processed/            # Preprocessed data
│   ├── models/              # Trained models
│   └── visualizations/      # Generated plots
├── config.yaml              # Configuration settings
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/sdb_project.git
cd sdb_project
```

### 2. Set Up Python Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv sdb_env

# Activate environment on Windows
sdb_env\Scripts\activate

# Activate environment on Linux/macOS
source sdb_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Using conda

```bash
# Create conda environment
conda create -n sdb_env python=3.9

# Activate environment
conda activate sdb_env

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test imports
python -c "import rasterio; import xgboost; import plotly; print('Installation successful!')"
```

## Auto-Processing All Regions 🚀

### **NEW: Batch Process All Regions with SAFE Files**

The project now includes an automated system that can process **all regions containing SAFE files** in a single command!

#### **Quick Start - Process All Regions**

```bash
# Process all regions automatically
python process_all_regions.py

# Show what would be processed (dry run)
python process_all_regions.py --dry-run

# Force reprocessing of already completed regions
python process_all_regions.py --force

# Process only a specific region
python process_all_regions.py --region goa
```

#### **What It Does**

1. **🔍 Discovers** all regions with SAFE files in `data/sentinel/{region}/raw/`
2. **🛰️ Auto-configures** the correct SAFE file for each region
3. **🔄 Runs the complete pipeline**: Band extraction → Preprocessing → Training → Visualization
4. **⏭️ Skips** already processed regions (unless `--force` used)
5. **📋 Logs** everything comprehensively
6. **🚀 Continues** processing even if individual regions fail

#### **Example Output**

```
🛰️ AUTO-PROCESS ALL SAFE REGIONS
==================================================

🔍 Found 3 regions with SAFE files:
   - goa
   - kachchh
   - palk_strait

--------------------------------------------------
PROCESSING SAFE REGIONS
--------------------------------------------------
[1/3] goa                   ✔ Completed
[2/3] kachchh               ✔ Completed
[3/3] palk_strait           ⏳ Processing
--------------------------------------------------

========= SUMMARY =========
Processed: 3 regions
Skipped:   0 regions
Failed:    0 regions
Outputs saved to: outputs/{region}/final_showcase/
============================
```

#### **Directory Structure After Processing**

```
sdb_project/
├── data/sentinel/
│   ├── goa/
│   │   ├── raw/               # Your SAFE files
│   │   └── processed/         # Auto-generated bands, arrays
│   ├── kachchh/
│   │   ├── raw/
│   │   └── processed/
│   └── palk_strait/
│       ├── raw/
│       └── processed/
├── outputs/
│   ├── goa/final_showcase/           # Visualizations
│   ├── kachchh/final_showcase/
│   └── palk_strait/final_showcase/
└── logs/
    └── process_all_regions_*.log     # Comprehensive logs
```

#### **Command Options**

| Option            | Description                                  |
| ----------------- | -------------------------------------------- |
| `--dry-run`       | Show what would be processed without running |
| `--force`         | Reprocess even if region already completed   |
| `--region <name>` | Process only the specified region            |
| `--help`          | Show all available options                   |

---

## Manual Region Processing

### **Individual Region Processing**

For processing a specific region manually:

```bash
# 1. Update coordinates in Streamlit interface
python run_region_selector.py

# 2. Or edit config directly
# Edit config/location_config.json with your coordinates

# 3. Run pipeline for current region
python run_full_pipeline.py
```

## Data Setup

### 1. Sentinel-2 Data

1. Register at [Copernicus Open Access Hub](https://scihub.copernicus.eu/)

   - Create an account if you don't have one
   - Save your credentials for later use

2. Create data directories:

```bash
mkdir -p data/sentinel
mkdir -p data/gebco_reference
```

3. Download Sentinel-2 Level-2A data:

#### Using the Download Script

```bash
python src/download_sentinel.py \
    --username YOUR_USERNAME \
    --password YOUR_PASSWORD \
    --bbox "longitude_min latitude_min longitude_max latitude_max" \
    --start-date "YYYY-MM-DD" \
    --end-date "YYYY-MM-DD" \
    --output data/sentinel
```

#### Manual Download

1. Visit [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
2. Search for Level-2A products in your area
3. Download required bands:
   - B02 (Blue, 490nm)
   - B03 (Green, 560nm)
   - B04 (Red, 665nm)
   - B08 (NIR, 842nm)
4. Extract to `data/sentinel/`

### 2. GEBCO Reference Data

1. Download bathymetry data:

   - Visit [GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)
   - Select your area of interest
   - Download the NetCDF grid file

2. Process GEBCO data:

```bash
# Extract points from GEBCO grid
python src/utils/extract_gebco_points.py \
    --input path/to/gebco.nc \
    --output data/gebco_reference/points.csv \
    --bbox "longitude_min latitude_min longitude_max latitude_max"
```

### 3. Area of Interest (Optional)

1. Create GeoJSON for your study area:
   - Use QGIS, geojson.io, or similar tools
   - Save as `data/aoi.geojson`
   - Format example:
     ```json
     {
       "type": "Polygon",
       "coordinates": [[[lon1,lat1], [lon2,lat2], ...]]
     }
     ```

## Usage

### 1. Configuration

Edit `config.yaml` to set your parameters:

```yaml
# Example configuration
output_dir: "output"
copernicus:
  username: "your_username"
  password: "your_password"
area:
  name: "test_area"
  bbox: [-123.0, 45.0, -122.0, 46.0]
  start_date: "2025-01-01"
  end_date: "2025-12-31"
```

### 2. Running the Pipeline

#### Command Line Interface

```bash
# Using configuration file
python src/main.py --config config.yaml

# Direct arguments
python src/main.py \
    --area "study_area" \
    --bbox "-123.0,45.0,-122.0,46.0" \
    --start-date "2025-01-01" \
    --end-date "2025-12-31" \
    --reference "data/gebco_reference/points.csv"
```

#### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/01_sdb_processing.ipynb
```

The notebook provides an interactive demonstration of:

- Data loading and preprocessing
- Model training and validation
- Results visualization and analysis

### 3. Output Structure

```
output/
├── processed/          # Preprocessed satellite data
│   ├── features/      # Extracted features
│   └── masks/         # Water and cloud masks
├── models/            # Trained ML models
│   ├── random_forest/ # Random Forest models
│   ├── xgboost/      # XGBoost models
│   └── svr/          # SVR models
└── visualizations/    # Generated plots
    ├── 2d/           # 2D bathymetry maps
    ├── 3d/           # 3D surface plots
    └── analysis/     # Error analysis plots
```

## Components

### Core Modules

#### `download_sentinel.py`

- Copernicus Hub authentication
- Scene search by area and date
- Parallel band download
- Cloud coverage filtering
- Automatic retry on failure

#### `preprocess.py`

- Atmospheric correction
- Cloud and shadow masking
- Water body detection
- Feature extraction:
  - Band ratios
  - Water indices
  - Spectral derivatives

#### `sdb_model.py`

- Multiple ML models:
  - Random Forest Regression
  - XGBoost
  - Support Vector Regression
- Cross-validation
- Feature importance analysis
- Model persistence
- Uncertainty estimation

#### `visualize.py`

- 2D bathymetry maps
- Interactive 3D surface plots
- Error distribution analysis
- Performance comparisons
- Custom colormaps
- GeoTIFF export

#### `main.py`

- Pipeline orchestration
- Configuration management
- Progress tracking
- Error handling
- Results export

## Extension Ideas

### 1. Data Sources

- Support for Landsat-8/9
- WorldView integration
- Wave/tide data incorporation
- Multiple date analysis

### 2. Processing

- Advanced atmospheric correction
- Multi-temporal analysis
- Wave effect compensation
- Tide state correction

### 3. Machine Learning

- Deep learning models (CNNs)
- Uncertainty quantification
- Transfer learning
- Ensemble methods

### 4. Applications

- Coastal change monitoring
- Habitat mapping
- Sediment transport
- Wave modeling integration

### 5. Visualization

- Web interface
- Time series animations
- GIS integration
- Real-time processing

## Contributing

1. Fork the repository
2. Create a feature branch:

```bash
git checkout -b feature/amazing-feature
```

3. Make your changes
4. Run tests:

```bash
python -m pytest tests/
```

5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```
@software{sdb_project,
  author = {Your Name},
  title = {Satellite-Derived Bathymetry Project},
  year = {2025},
  url = {https://github.com/yourusername/sdb_project}
}
```
