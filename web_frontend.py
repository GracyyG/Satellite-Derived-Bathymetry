#!/usr/bin/env python3
"""
Modern Web Frontend for SDB Project
Flask-based web interface for interactive satellite bathymetry visualization
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"
STATIC_DIR = PROJECT_ROOT / "web" / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create Flask app with proper template directory
app = Flask(__name__, 
           template_folder=str(TEMPLATES_DIR),
           static_folder=str(STATIC_DIR))
app.secret_key = 'sdb_project_2025'

class SDBWebInterface:
    """Main web interface controller"""
    
    def __init__(self):
        self.current_status = {"status": "idle", "message": "", "progress": 0}
        self.available_regions = self.scan_available_regions()
    
    def scan_available_regions(self):
        """Scan for available regions with data and results"""
        regions = {}
        
        # Predefined regions
        predefined = {
            "lakshadweep": {"name": "Lakshadweep", "lat_range": [10.75, 10.95], "lon_range": [72.35, 72.65]},
            "goa": {"name": "Goa", "lat_range": [15.0, 15.8], "lon_range": [73.5, 74.5]},
            "kachchh": {"name": "Kachchh", "lat_range": [22.5, 23.5], "lon_range": [68.5, 70.0]},
            "palk_strait": {"name": "Palk Strait", "lat_range": [9.0, 10.5], "lon_range": [78.5, 80.0]},
        }
        
        for region_id, info in predefined.items():
            region_data = {
                "id": region_id,
                "name": info["name"],
                "lat_range": info["lat_range"],
                "lon_range": info["lon_range"],
                "has_data": False,
                "has_models": False,
                "has_results": False,
                "result_files": []
            }
            
            # Check for data
            data_path = PROJECT_ROOT / "data" / "processed" / region_id / "training_data"
            if data_path.exists():
                region_data["has_data"] = True
            
            # Check for models
            models_path = PROJECT_ROOT / "models" / region_id
            if models_path.exists() and list(models_path.glob("*.joblib")):
                region_data["has_models"] = True
            
            # Check for results
            results_path = OUTPUTS_DIR / region_id
            if results_path.exists():
                region_data["has_results"] = True
                
                # Scan for visualization files
                viz_files = []
                for pattern in ["*.html", "*.png", "*.jpg", "*.json"]:
                    viz_files.extend(list(results_path.rglob(pattern)))
                
                region_data["result_files"] = [
                    {
                        "name": f.name,
                        "path": str(f.relative_to(PROJECT_ROOT)),
                        "type": f.suffix[1:],
                        "size": f.stat().st_size if f.exists() else 0,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if f.exists() else ""
                    }
                    for f in sorted(viz_files)[-20:]  # Latest 20 files
                ]
            
            regions[region_id] = region_data
        
        return regions
    
    def get_region_summary(self, region_id):
        """Get detailed summary for a specific region"""
        if region_id not in self.available_regions:
            return None
        
        region = self.available_regions[region_id]
        
        # Get model metrics if available
        metrics_path = PROJECT_ROOT / "models" / region_id / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except:
                pass
        
        # Get latest results
        latest_results = []
        results_path = OUTPUTS_DIR / region_id
        if results_path.exists():
            # Find key result files
            key_patterns = [
                "*showcase*",
                "*comparison*", 
                "*3d*",
                "*comprehensive*",
                "*performance*"
            ]
            
            for pattern in key_patterns:
                files = list(results_path.rglob(pattern))
                latest_results.extend(files)
        
        return {
            **region,
            "metrics": metrics,
            "latest_results": [f.relative_to(PROJECT_ROOT) for f in latest_results]
        }

# Initialize web interface
web_interface = SDBWebInterface()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html', regions=web_interface.available_regions)

@app.route('/region/<region_id>')
def region_detail(region_id):
    """Region detail page"""
    region_data = web_interface.get_region_summary(region_id)
    if not region_data:
        return redirect(url_for('index'))
    
    return render_template('region_detail.html', region=region_data)

@app.route('/api/regions')
def api_regions():
    """API endpoint for region data"""
    return jsonify(web_interface.available_regions)

@app.route('/api/region/<region_id>')
def api_region_detail(region_id):
    """API endpoint for specific region"""
    region_data = web_interface.get_region_summary(region_id)
    if not region_data:
        return jsonify({"error": "Region not found"}), 404
    return jsonify(region_data)

@app.route('/api/status')
def api_status():
    """Get current processing status"""
    return jsonify(web_interface.current_status)

@app.route('/api/run_pipeline/<region_id>', methods=['POST'])
def api_run_pipeline(region_id):
    """Run pipeline for specific region"""
    
    def run_pipeline_thread():
        web_interface.current_status = {
            "status": "running",
            "message": f"Processing {region_id}...",
            "progress": 10
        }
        
        try:
            # Update region config
            config_path = PROJECT_ROOT / "config" / "location_config.json"
            region_info = web_interface.available_regions.get(region_id, {})
            
            if region_info:
                config = {
                    "region_name": region_id,
                    "aoi": {
                        "min_lat": region_info["lat_range"][0],
                        "max_lat": region_info["lat_range"][1],
                        "min_lon": region_info["lon_range"][0],
                        "max_lon": region_info["lon_range"][1]
                    }
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                web_interface.current_status["progress"] = 30
                web_interface.current_status["message"] = "Configuration updated, running visualizations..."
            
            # Run ONLY the visualization script (30 seconds max)
            viz_script = PROJECT_ROOT / "visualisations" / "comprehensive_model_comparison.py"
            if viz_script.exists():
                web_interface.current_status["progress"] = 60
                web_interface.current_status["message"] = "Generating visualizations from existing models..."
                
                result = subprocess.run([
                    "python", str(viz_script), "--region", region_id
                ], cwd=PROJECT_ROOT, capture_output=True, text=True)
                
                web_interface.current_status["progress"] = 100
                
                if result.returncode == 0:
                    web_interface.current_status = {
                        "status": "completed",
                        "message": f"Successfully generated visualizations for {region_id}",
                        "progress": 100
                    }
                else:
                    web_interface.current_status = {
                        "status": "error",
                        "message": f"Visualization error: {result.stderr}",
                        "progress": 0
                    }
            else:
                web_interface.current_status = {
                    "status": "error", 
                    "message": "Visualization script not found",
                    "progress": 0
                }
                
        except Exception as e:
            web_interface.current_status = {
                "status": "error",
                "message": f"Pipeline error: {str(e)}",
                "progress": 0
            }
        
        # Refresh region data
        web_interface.available_regions = web_interface.scan_available_regions()
    
    # Run in background thread
    thread = threading.Thread(target=run_pipeline_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Pipeline started", "status": "started"})

@app.route('/files/<path:filepath>')
def serve_file(filepath):
    """Serve result files"""
    file_path = PROJECT_ROOT / filepath
    if file_path.exists() and file_path.is_file():
        return send_from_directory(file_path.parent, file_path.name)
    return "File not found", 404

@app.route('/api/experiments')
def api_experiments():
    """Get cross-region experiment results"""
    experiments_path = PROJECT_ROOT / "experiments" / "cross_region_transfer"
    experiments = []
    
    if experiments_path.exists():
        # Regular results
        results_path = experiments_path / "results"
        if results_path.exists():
            for f in results_path.glob("*.html"):
                experiments.append({
                    "name": f.stem,
                    "type": "transfer_analysis",
                    "path": str(f.relative_to(PROJECT_ROOT)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
        
        # 3D plots
        plots_3d_path = experiments_path / "3d_plots"
        if plots_3d_path.exists():
            for f in plots_3d_path.glob("*.html"):
                experiments.append({
                    "name": f.stem,
                    "type": "3d_visualization",
                    "path": str(f.relative_to(PROJECT_ROOT)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    
    return jsonify(experiments)

def create_templates():
    """Create HTML templates"""
    
    # Ensure template directory exists
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating templates in: {TEMPLATES_DIR}")
    print(f"Creating static files in: {STATIC_DIR}")
    
    # Create main template - Modern, minimalistic design
    index_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SDB · Satellite Bathymetry</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f0f;
            color: #e8e8e8;
            font-weight: 400;
            line-height: 1.6;
        }
        
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 0 24px; 
        }
        
        /* Header */
        .header {
            border-bottom: 1px solid #1f1f1f;
            background: rgba(15, 15, 15, 0.9);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 0;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-indicator {
            height: 8px;
            width: 8px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Main Content */
        .main {
            padding: 40px 0;
        }
        
        .hero {
            text-align: center;
            margin-bottom: 60px;
        }
        
        .hero h1 {
            font-size: 48px;
            font-weight: 300;
            color: #fff;
            margin-bottom: 16px;
            letter-spacing: -0.02em;
        }
        
        .hero p {
            font-size: 18px;
            color: #888;
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Region Form */
        .region-form {
            background: #171717;
            border: 1px solid #2a2a2a;
            border-radius: 16px;
            padding: 32px;
            margin: 40px auto 60px;
            max-width: 800px;
        }
        
        .form-header {
            text-align: center;
            margin-bottom: 32px;
        }
        
        .form-header h2 {
            color: #fff;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .form-header p {
            color: #666;
            font-size: 16px;
        }
        
        .form-content {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group label {
            color: #fff;
            font-weight: 500;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .form-input, .form-select {
            background: #0f0f0f;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 12px 16px;
            color: #fff;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .form-input:focus, .form-select:focus {
            outline: none;
            border-color: #0088ff;
            box-shadow: 0 0 0 3px rgba(0, 136, 255, 0.1);
        }
        
        .form-input::placeholder {
            color: #555;
        }
        
        .form-select option {
            background: #171717;
            color: #fff;
        }
        
        .form-actions {
            display: flex;
            gap: 16px;
            justify-content: center;
            margin-top: 16px;
        }
        
        .btn {
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: #0088ff;
            color: #fff;
        }
        
        .btn-primary:hover {
            background: #0077ee;
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background: #2a2a2a;
            color: #e8e8e8;
            border: 1px solid #404040;
        }
        
        .btn-secondary:hover {
            background: #353535;
            border-color: #505050;
        }
        

        
        /* Status Panel */
        .status-panel {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid #333;
            border-radius: 16px;
            padding: 24px;
            z-index: 1000;
            transform: translateX(120%);
            transition: all 0.4s ease;
            min-width: 320px;
            backdrop-filter: blur(10px);
        }
        
        .status-panel.active {
            transform: translateX(0);
        }
        
        .status-content {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .progress-circle {
            position: relative;
            width: 60px;
            height: 60px;
        }
        
        .progress-ring {
            width: 60px;
            height: 60px;
            transform: rotate(-90deg);
        }
        
        .progress-ring-bg {
            stroke-dasharray: 157;
            stroke-dashoffset: 0;
        }
        
        .progress-ring-progress {
            stroke-dasharray: 157;
            stroke-dashoffset: 157;
            transition: stroke-dashoffset 0.3s ease;
        }
        
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #0088ff;
            font-size: 12px;
            font-weight: 600;
        }
        
        .status-info {
            flex: 1;
        }
        
        .status-message {
            color: #fff;
            font-weight: 500;
            font-size: 14px;
            margin-bottom: 4px;
        }
        
        .time-remaining {
            color: #888;
            font-size: 12px;
        }
        
        /* Completion Notification */
        .completion-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 136, 0, 0.95);
            border: 1px solid #00aa00;
            border-radius: 16px;
            padding: 20px;
            z-index: 1001;
            transform: translateX(120%);
            transition: all 0.4s ease;
            min-width: 320px;
            backdrop-filter: blur(10px);
        }
        
        .completion-notification.show {
            transform: translateX(0);
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .success-icon {
            font-size: 24px;
        }
        
        .notification-text {
            flex: 1;
        }
        
        .notification-title {
            color: #fff;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }
        
        .notification-subtitle {
            color: #ccffcc;
            font-size: 12px;
        }
        
        .notification-close {
            background: none;
            border: none;
            color: #fff;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: background-color 0.2s ease;
        }
        
        .notification-close:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 { font-size: 36px; }
            .regions-grid { grid-template-columns: 1fr; }
            .region-actions { grid-template-columns: 1fr; }
            .experiment-actions { flex-direction: column; }
        }
        

    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <div class="header-content">
                <a href="/" class="logo">
                    <div class="status-indicator"></div>
                    SDB
                </a>
                <div style="font-size: 14px; color: #666;">Satellite Bathymetry Interface</div>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="container">
            <div class="hero">
                <h1>Coastal Water Mapping</h1>
                <p>Advanced satellite-derived bathymetry using machine learning for precise coastal water depth estimation</p>
            </div>
            
            <!-- Status Panel -->
            <div id="statusPanel" class="status-panel">
                <div class="status-content">
                    <div class="progress-circle" id="progressCircle">
                        <svg class="progress-ring" width="60" height="60">
                            <circle class="progress-ring-bg" cx="30" cy="30" r="25" fill="transparent" stroke="#333" stroke-width="4"/>
                            <circle class="progress-ring-progress" cx="30" cy="30" r="25" fill="transparent" stroke="#0088ff" stroke-width="4" stroke-linecap="round" transform="rotate(-90 30 30)"/>
                        </svg>
                        <div class="progress-text" id="progressText">0%</div>
                    </div>
                    <div class="status-info">
                        <div id="statusMessage" class="status-message">Processing...</div>
                        <div id="timeRemaining" class="time-remaining">Estimating time...</div>
                    </div>
                </div>
            </div>
            
            <!-- Completion Notification -->
            <div id="completionNotification" class="completion-notification">
                <div class="notification-content">
                    <div class="success-icon">✅</div>
                    <div class="notification-text">
                        <div class="notification-title">Visualizations Ready!</div>
                        <div class="notification-subtitle">Click "View Results" to explore your bathymetry analysis</div>
                    </div>
                    <button class="notification-close" onclick="hideCompletionNotification()">×</button>
                </div>
            </div>
            
            <!-- Region Selection Form -->
            <div class="region-form">
                <div class="form-header">
                    <h2>Select Region</h2>
                    <p>Choose from saved regions or define custom coordinates</p>
                </div>
                
                <div class="form-content">
                    <div class="form-group">
                        <label>Saved Regions</label>
                        <select id="savedRegions" class="form-select" onchange="loadSavedRegion()">
                            <option value="">Select a saved region...</option>
                            {% for region_id, region in regions.items() %}
                                {% if region.has_results %}
                                <option value="{{ region_id }}" data-name="{{ region.name }}" 
                                        data-minlat="{{ region.lat_range[0] }}" data-maxlat="{{ region.lat_range[1] }}"
                                        data-minlon="{{ region.lon_range[0] }}" data-maxlon="{{ region.lon_range[1] }}">
                                    {{ region.name }}
                                </option>
                                {% endif %}
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label>Region Name</label>
                            <input type="text" id="regionName" class="form-input" placeholder="e.g., Lakshadweep">
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label>Min Latitude</label>
                            <input type="number" id="minLat" class="form-input" step="0.0001" placeholder="10.7500">
                        </div>
                        <div class="form-group">
                            <label>Max Latitude</label>
                            <input type="number" id="maxLat" class="form-input" step="0.0001" placeholder="10.9500">
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label>Min Longitude</label>
                            <input type="number" id="minLon" class="form-input" step="0.0001" placeholder="72.3500">
                        </div>
                        <div class="form-group">
                            <label>Max Longitude</label>
                            <input type="number" id="maxLon" class="form-input" step="0.0001" placeholder="72.6500">
                        </div>
                    </div>
                    
                    <div class="form-actions">
                        <button class="btn btn-primary" onclick="viewResults()">View Results</button>
                        <button class="btn btn-secondary" onclick="generateVisualizations()">Generate Visualizations</button>
                    </div>
                </div>
            </div>
            

        </div>
    </div>

    <script>
        let statusCheckInterval;
        
        function runPipeline(regionId) {
            const panel = document.getElementById('statusPanel');
            panel.className = 'status-panel active';
            document.getElementById('statusMessage').textContent = `Generating visualizations for ${regionId}...`;
            document.getElementById('statusProgress').style.width = '10%';
            
            fetch(`/api/run_pipeline/${regionId}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        statusCheckInterval = setInterval(checkStatus, 2000);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('statusMessage').textContent = 'Error starting pipeline';
                    panel.className = 'status-panel active error';
                });
        }
        
        function checkStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const panel = document.getElementById('statusPanel');
                    document.getElementById('statusMessage').textContent = data.message;
                    updateProgressCircle(data.progress);
                    updateTimeRemaining(data.progress);
                    
                    if (data.status === 'completed') {
                        clearInterval(statusCheckInterval);
                        updateProgressCircle(100);
                        document.getElementById('statusMessage').textContent = 'Visualizations completed!';
                        document.getElementById('timeRemaining').textContent = 'Ready to view';
                        
                        setTimeout(() => {
                            panel.className = 'status-panel';
                            showCompletionNotification();
                        }, 1500);
                    } else if (data.status === 'error') {
                        clearInterval(statusCheckInterval);
                        updateProgressCircle(0);
                        document.getElementById('timeRemaining').textContent = 'Process failed';
                        setTimeout(() => panel.className = 'status-panel', 5000);
                    }
                })
                .catch(error => {
                    console.error('Status check error:', error);
                    clearInterval(statusCheckInterval);
                    document.getElementById('statusPanel').className = 'status-panel';
                });
        }
        

        
        function view3DPlots() {
            const plots = [
                'experiments/cross_region_transfer/3d_plots/transfer_comprehensive_3d_showcase.html',
                'experiments/cross_region_transfer/3d_plots/transfer_3d_bathymetry_surfaces.html'
            ];
            
            plots.forEach(plot => window.open(`/files/${plot}`, '_blank'));
        }
        
        function loadSavedRegion() {
            const select = document.getElementById('savedRegions');
            const option = select.options[select.selectedIndex];
            
            if (option.value) {
                document.getElementById('regionName').value = option.getAttribute('data-name');
                document.getElementById('minLat').value = option.getAttribute('data-minlat');
                document.getElementById('maxLat').value = option.getAttribute('data-maxlat');
                document.getElementById('minLon').value = option.getAttribute('data-minlon');
                document.getElementById('maxLon').value = option.getAttribute('data-maxlon');
            }
        }
        
        function viewResults() {
            const regionName = document.getElementById('regionName').value;
            const select = document.getElementById('savedRegions');
            
            if (select.value) {
                window.location.href = '/region/' + select.value;
            } else if (regionName) {
                const regionId = regionName.toLowerCase().replace(/\\s+/g, '_');
                window.location.href = '/region/' + regionId;
            } else {
                alert('Please select a saved region or enter a region name');
            }
        }
        
        let startTime = null;
        let estimatedDuration = 300; // 5 minutes default
        
        function updateProgressCircle(percent) {
            const circle = document.querySelector('.progress-ring-progress');
            const circumference = 2 * Math.PI * 25; // radius = 25
            
            // Ensure percent is a valid number between 0 and 100
            const validPercent = Math.max(0, Math.min(100, isNaN(percent) ? 0 : percent));
            
            const offset = circumference - (validPercent / 100) * circumference;
            circle.style.strokeDasharray = circumference;
            circle.style.strokeDashoffset = offset;
            document.getElementById('progressText').textContent = Math.round(validPercent) + '%';
        }
        
        function updateTimeRemaining(progress) {
            if (!startTime || progress <= 0) {
                document.getElementById('timeRemaining').textContent = 'Starting...';
                return;
            }
            
            const elapsed = (Date.now() - startTime) / 1000;
            const totalEstimated = elapsed / (progress / 100);
            const remaining = Math.max(0, totalEstimated - elapsed);
            
            // Check for invalid calculations
            if (!isFinite(remaining) || isNaN(remaining)) {
                document.getElementById('timeRemaining').textContent = 'Processing...';
                return;
            }
            
            const minutes = Math.floor(remaining / 60);
            const seconds = Math.floor(remaining % 60);
            
            if (remaining > 0 && remaining < 3600) { // Cap at 1 hour max
                document.getElementById('timeRemaining').textContent = 
                    `${minutes}m ${seconds}s remaining`;
            } else {
                document.getElementById('timeRemaining').textContent = 'Almost done...';
            }
        }
        
        function showCompletionNotification() {
            document.getElementById('completionNotification').className = 'completion-notification show';
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                hideCompletionNotification();
            }, 10000);
        }
        
        function hideCompletionNotification() {
            document.getElementById('completionNotification').className = 'completion-notification';
        }
        
        function generateVisualizations() {
            const regionName = document.getElementById('regionName').value;
            const minLat = document.getElementById('minLat').value;
            const maxLat = document.getElementById('maxLat').value;
            const minLon = document.getElementById('minLon').value;
            const maxLon = document.getElementById('maxLon').value;
            
            if (!regionName || !minLat || !maxLat || !minLon || !maxLon) {
                alert('Please fill in all fields');
                return;
            }
            
            const regionId = regionName.toLowerCase().replace(/\\s+/g, '_');
            const panel = document.getElementById('statusPanel');
            
            // Initialize progress tracking
            startTime = Date.now();
            panel.className = 'status-panel active';
            document.getElementById('statusMessage').textContent = `Generating visualizations for ${regionName}...`;
            updateProgressCircle(5);
            document.getElementById('timeRemaining').textContent = 'Estimating time...';
            
            fetch(`/api/run_pipeline/${regionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: regionName,
                    lat_range: [parseFloat(minLat), parseFloat(maxLat)],
                    lon_range: [parseFloat(minLon), parseFloat(maxLon)]
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    statusCheckInterval = setInterval(checkStatus, 2000);
                } else {
                    document.getElementById('statusMessage').textContent = 'Error: ' + (data.error || 'Unknown error');
                    panel.className = 'status-panel active';
                    updateProgressCircle(0);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('statusMessage').textContent = 'Error starting pipeline';
                panel.className = 'status-panel active';
                updateProgressCircle(0);
            });
        }
    </script>
</body>
</html>"""
    
    # Save index template
    with open(TEMPLATES_DIR / "index.html", 'w', encoding='utf-8') as f:
        f.write(index_template)
    
    # Create region detail template - Modern, minimalistic design
    region_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ region.name }} · SDB Results</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f0f;
            color: #e8e8e8;
            font-weight: 400;
            line-height: 1.6;
        }
        
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 0 24px; 
        }
        
        /* Header */
        .header {
            border-bottom: 1px solid #1f1f1f;
            background: rgba(15, 15, 15, 0.9);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 0;
        }
        
        .back-btn {
            padding: 8px 16px;
            background: #2a2a2a;
            border: 1px solid #404040;
            border-radius: 8px;
            color: #e8e8e8;
            text-decoration: none;
            font-size: 14px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .back-btn:hover {
            background: #353535;
            color: #fff;
        }
        
        .region-title {
            font-size: 32px;
            font-weight: 300;
            color: #fff;
        }
        
        /* Main */
        .main {
            padding: 40px 0;
        }
        
        /* Info Section */
        .info-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 60px;
        }
        
        .info-card {
            background: #171717;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 24px;
        }
        
        .info-card h3 {
            color: #fff;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
        }
        
        .coords {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            color: #888;
            line-height: 1.8;
        }
        
        .metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }
        
        .metric-badge {
            padding: 6px 12px;
            background: #00ff8814;
            color: #00ff88;
            border: 1px solid #00ff8820;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .status-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .badge {
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        
        .badge-success { background: #00ff8814; color: #00ff88; border: 1px solid #00ff8820; }
        .badge-danger { background: #ff475714; color: #ff4757; border: 1px solid #ff475720; }
        .badge-info { background: #0088ff14; color: #0088ff; border: 1px solid #0088ff20; }
        
        /* Gallery Section */
        .gallery-section {
            margin-bottom: 40px;
        }
        
        .gallery-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }
        
        .gallery-header h2 {
            font-size: 24px;
            font-weight: 600;
            color: #fff;
        }
        
        .gallery-count {
            color: #666;
            font-size: 14px;
            background: #171717;
            padding: 6px 12px;
            border-radius: 20px;
            border: 1px solid #2a2a2a;
        }
        
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
        }
        
        .gallery-item {
            background: #171717;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 16px;
            transition: all 0.3s ease;
        }
        
        .gallery-item:hover {
            border-color: #404040;
            transform: translateY(-2px);
        }
        
        .item-preview {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }
        
        .preview-icon {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        
        .preview-icon.interactive { background: #0088ff14; }
        .preview-icon.image { background: #00ff8814; }
        .preview-icon.data { background: #ffa50014; }
        
        .item-type {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #666;
            font-weight: 500;
        }
        
        .item-details {
            flex: 1;
            min-width: 0;
        }
        
        .item-name {
            color: #fff;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .item-size {
            color: #666;
            font-size: 12px;
        }
        
        .item-action {
            padding: 8px 12px;
            background: #0088ff;
            color: #fff;
            text-decoration: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .item-action:hover {
            background: #0077ee;
            color: #fff;
        }
        
        /* Empty Gallery */
        .empty-gallery {
            text-align: center;
            padding: 60px 20px;
            background: #171717;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
        }
        
        .empty-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        .empty-gallery h3 {
            color: #fff;
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        .empty-gallery p {
            color: #666;
            font-size: 14px;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .info-section { 
                grid-template-columns: 1fr; 
            }
            
            .gallery-grid {
                grid-template-columns: 1fr;
            }
            
            .gallery-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            
            .form-row {
                grid-template-columns: 1fr;
            }
            
            .form-actions {
                flex-direction: column;
            }
            
            .region-title {
                font-size: 24px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <div class="header-content">
                <h1 class="region-title">{{ region.name }}</h1>
                <a href="/" class="back-btn">
                    ← Back to Dashboard
                </a>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="container">
            <!-- Info Section -->
            <div class="info-section">
                <div class="info-card">
                    <h3>Region Information</h3>
                    <div class="coords">
                        Latitude: {{ "%.4f"|format(region.lat_range[0]) }}° to {{ "%.4f"|format(region.lat_range[1]) }}°N<br>
                        Longitude: {{ "%.4f"|format(region.lon_range[0]) }}° to {{ "%.4f"|format(region.lon_range[1]) }}°E
                    </div>
                    
                    {% if region.metrics %}
                    <div class="metrics">
                        {% for model, metrics in region.metrics.items() %}
                            {% if 'rmse' in metrics %}
                            <div class="metric-badge">
                                {{ model.replace('_', ' ').title() }}: {{ "%.3f"|format(metrics.rmse) }}m
                            </div>
                            {% endif %}
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
                
                <div class="info-card">
                    <h3>Region Status</h3>
                    <div class="status-badges">
                        {% if region.has_data %}
                            <span class="badge badge-success">Training Data Available</span>
                        {% endif %}
                        
                        {% if region.has_models %}
                            <span class="badge badge-success">Models Trained</span>
                        {% endif %}
                        
                        {% if region.result_files|length > 0 %}
                            <span class="badge badge-info">{{ region.result_files|length }} Visualizations</span>
                        {% endif %}
                        
                        {% if not (region.has_data or region.has_models or region.result_files|length > 0) %}
                            <span class="badge badge-info">Ready for Processing</span>
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <!-- Results Gallery -->
            <div class="gallery-section">
                <div class="gallery-header">
                    <h2>Visualizations</h2>
                    <div class="gallery-count">{{ region.result_files|length }} files</div>
                </div>
                
                {% if region.result_files %}
                <div class="gallery-grid">
                    {% for file in region.result_files %}
                    <div class="gallery-item">
                        <div class="item-preview">
                            {% if file.type == 'html' %}
                                <div class="preview-icon interactive">🌐</div>
                                <div class="item-type">Interactive</div>
                            {% elif file.type in ['png', 'jpg', 'jpeg'] %}
                                <div class="preview-icon image">📊</div>
                                <div class="item-type">Chart</div>
                            {% else %}
                                <div class="preview-icon data">📄</div>
                                <div class="item-type">Data</div>
                            {% endif %}
                        </div>
                        <div class="item-details">
                            <div class="item-name">{{ file.name.replace('_', ' ').replace('.html', '').replace('.png', '').replace('.json', '') | title }}</div>
                            <div class="item-size">{{ (file.size / 1024) | round | int }} KB</div>
                        </div>
                        <a href="/files/{{ file.path }}" target="_blank" class="item-action">View</a>
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <div class="empty-gallery">
                    <div class="empty-icon">📊</div>
                    <h3>No Visualizations Yet</h3>
                    <p>Generate visualizations from the main dashboard to populate this gallery</p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>"""
    
    # Save region template
    with open(TEMPLATES_DIR / "region_detail.html", 'w', encoding='utf-8') as f:
        f.write(region_template)

if __name__ == "__main__":
    print("🌊 SDB Web Interface Starting...")
    print("="*50)
    print("📱 Modern Web Frontend for Satellite Bathymetry")
    print("🔗 Access at: http://localhost:5000")
    print("✨ Features:")
    print("   • Interactive region selection")
    print("   • Real-time pipeline execution")
    print("   • Visualization gallery")
    print("   • 3D plot integration")
    print("   • Cross-region experiments")
    print("="*50)
    
    # Create templates before starting server
    create_templates()
    
    app.run(debug=True, host='0.0.0.0', port=5000)