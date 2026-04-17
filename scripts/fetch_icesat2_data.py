#!/usr/bin/env python3
"""
ICESat-2 Data Fetcher for Kachchh Region
Fetches real bathymetry data from NASA Earthdata and maps it to Sentinel-2 grid
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import h5py
from shapely.geometry import Point, Polygon
import argparse
import time

class ICESat2Fetcher:
    """Fetches ICESat-2 ATL03 data for bathymetry analysis"""
    
    def __init__(self, config_path="config/icesat_config.json"):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / config_path
        self.load_credentials()
        self.session = None
        
    def load_credentials(self):
        """Load ICESat-2 credentials"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"ICESat config not found: {self.config_path}")
            
        with open(self.config_path) as f:
            self.config = json.load(f)
            
        self.username = self.config['username']
        self.password = self.config['password']
        print(f"[OK] Loaded ICESat-2 credentials for: {self.username}")
    
    def authenticate(self):
        """Authenticate with NASA Earthdata"""
        self.session = requests.Session()
        
        # NASA Earthdata login
        login_url = "https://urs.earthdata.nasa.gov/oauth/authorize"
        
        # Create session and authenticate
        auth_response = self.session.post(
            "https://urs.earthdata.nasa.gov/login",
            data={
                'username': self.username,
                'password': self.password
            }
        )
        
        if auth_response.status_code == 200:
            print("[OK] Successfully authenticated with NASA Earthdata")
            return True
        else:
            print(f"[ERROR] Authentication failed: {auth_response.status_code}")
            return False
    
    def search_granules(self, bbox, start_date, end_date):
        """Search for ICESat-2 ATL03 granules in the AOI"""
        
        # CMR (Common Metadata Repository) search
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        # Expand search area by 0.5 degrees for better coverage
        expanded_bbox = {
            'min_lat': bbox['min_lat'] - 0.5,
            'max_lat': bbox['max_lat'] + 0.5,
            'min_lon': bbox['min_lon'] - 0.5,
            'max_lon': bbox['max_lon'] + 0.5
        }
        
        # Try multiple dataset versions
        concept_ids = [
            'C1511847675-NSIDC_ECS',  # ATL03 v005
            'C2153572614-NSIDC_ECS',  # ATL03 v006
        ]
        
        all_granules = []
        
        for concept_id in concept_ids:
            params = {
                'concept_id': concept_id,
                'bounding_box': f"{expanded_bbox['min_lon']},{expanded_bbox['min_lat']},{expanded_bbox['max_lon']},{expanded_bbox['max_lat']}",
                'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
                'page_size': 100
            }
            
            print(f"[INFO] Searching ICESat-2 granules (concept: {concept_id})...")
            print(f"  Expanded AOI: {expanded_bbox}")
            print(f"  Date range: {start_date} to {end_date}")
            
            response = requests.get(cmr_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                granules = data.get('feed', {}).get('entry', [])
                all_granules.extend(granules)
                print(f"  Found {len(granules)} granules for {concept_id}")
            else:
                print(f"  Search failed for {concept_id}: {response.status_code}")
        
        print(f"[OK] Total granules found: {len(all_granules)}")
        
        # If still no granules, try a simpler search without bounding box
        if not all_granules:
            print("[INFO] Trying region-based search...")
            
            params = {
                'concept_id': 'C1511847675-NSIDC_ECS',
                'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
                'page_size': 50
            }
            
            response = requests.get(cmr_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                all_granules_global = data.get('feed', {}).get('entry', [])
                print(f"[INFO] Found {len(all_granules_global)} granules globally, filtering by region...")
                
                # Filter by approximate region
                for granule in all_granules_global[:20]:  # Check first 20
                    title = granule.get('title', '')
                    # Basic filtering by title patterns that might indicate coverage area
                    if any(region in title.lower() for region in ['asia', 'indian', 'arabia']):
                        all_granules.append(granule)
                
                print(f"[INFO] Filtered to {len(all_granules)} potentially relevant granules")
        
        return all_granules
    
    def download_granule(self, granule_url, output_path):
        """Download a single ICESat-2 granule"""
        
        if not self.session:
            if not self.authenticate():
                return False
                
        print(f"[INFO] Downloading: {Path(granule_url).name}")
        
        response = self.session.get(granule_url, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"[ERROR] Download failed: {response.status_code}")
            return False
    
    def extract_bathymetry_points(self, h5_file_path, bbox):
        """Extract bathymetry points from ATL03 HDF5 file"""
        
        points = []
        
        try:
            with h5py.File(h5_file_path, 'r') as f:
                
                # ATL03 has 6 beams: gt1l, gt1r, gt2l, gt2r, gt3l, gt3r
                beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
                
                for beam in beams:
                    if beam not in f:
                        continue
                        
                    try:
                        # Get photon data
                        heights_group = f[f'{beam}/heights']
                        
                        if 'lat_ph' not in heights_group or 'lon_ph' not in heights_group:
                            continue
                            
                        lats = heights_group['lat_ph'][:]
                        lons = heights_group['lon_ph'][:]
                        heights = heights_group['h_ph'][:]
                        
                        # Quality flags (keep only good quality photons)
                        if 'signal_conf_ph' in heights_group:
                            quality = heights_group['signal_conf_ph'][:]
                            # Keep high confidence photons (quality >= 3)
                            mask = quality >= 3
                        else:
                            mask = np.ones(len(lats), dtype=bool)
                        
                        # Filter by AOI bounds
                        in_aoi = (
                            (lats >= bbox['min_lat']) & (lats <= bbox['max_lat']) &
                            (lons >= bbox['min_lon']) & (lons <= bbox['max_lon'])
                        )
                        
                        final_mask = mask & in_aoi
                        
                        if np.any(final_mask):
                            beam_points = pd.DataFrame({
                                'lat': lats[final_mask],
                                'lon': lons[final_mask],
                                'elevation': heights[final_mask],
                                'beam': beam
                            })
                            
                            # Filter for likely bathymetry points (negative elevations near coast)
                            # Keep points between -50m and +2m (to include some land reference)
                            bathy_mask = (beam_points['elevation'] >= -50) & (beam_points['elevation'] <= 2)
                            beam_points = beam_points[bathy_mask]
                            
                            if not beam_points.empty:
                                points.append(beam_points)
                                print(f"    {beam}: {len(beam_points)} bathymetry points")
                    
                    except Exception as e:
                        print(f"[WARN] Error processing beam {beam}: {e}")
                        continue
                        
        except Exception as e:
            print(f"[ERROR] Failed to read HDF5 file: {e}")
            return pd.DataFrame()
        
        if points:
            all_points = pd.concat(points, ignore_index=True)
            print(f"[OK] Extracted {len(all_points)} total bathymetry points")
            return all_points
        else:
            return pd.DataFrame()

def fetch_icesat2_data(region_name, start_date="2022-01-01", end_date="2024-12-31"):
    """Main function to fetch ICESat-2 data for a region"""
    
    # Load AOI from config
    config_path = Path("config/location_config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    aoi = config['aoi']
    
    print("="*80)
    print(f"FETCHING ICESAT-2 DATA FOR REGION: {region_name.upper()}")
    print("="*80)
    print(f"AOI: {aoi}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Initialize fetcher
    fetcher = ICESat2Fetcher()
    
    # Create data directory
    data_dir = Path(f"data/icesat2/{region_name}/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for granules
    granules = fetcher.search_granules(aoi, start_date, end_date)
    
    if not granules:
        print("[ERROR] No ICESat-2 granules found for this region/time period")
        return None
    
    # Download and process granules (limit to first 5 for demo)
    all_points = []
    
    for i, granule in enumerate(granules[:5]):  # Limit downloads
        
        # Get download URL
        download_url = None
        for link in granule.get('links', []):
            if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                download_url = link['href']
                break
        
        if not download_url:
            continue
            
        # Download file
        filename = granule['title'] + '.h5'
        file_path = data_dir / filename
        
        if not file_path.exists():
            if not fetcher.download_granule(download_url, file_path):
                continue
        
        # Extract bathymetry points
        points_df = fetcher.extract_bathymetry_points(file_path, aoi)
        
        if not points_df.empty:
            all_points.append(points_df)
        
        # Clean up large file
        if file_path.exists() and file_path.stat().st_size > 100_000_000:  # > 100MB
            file_path.unlink()  # Delete to save space
    
    if not all_points:
        print("[ERROR] No bathymetry points extracted from granules")
        return None
    
    # Combine all points
    combined_points = pd.concat(all_points, ignore_index=True)
    
    # Save raw points
    points_file = data_dir.parent / "bathymetry_points.csv"
    combined_points.to_csv(points_file, index=False)
    
    print(f"\n[OK] ICESat-2 data fetch complete!")
    print(f"Total points: {len(combined_points)}")
    print(f"Elevation range: {combined_points['elevation'].min():.1f}m to {combined_points['elevation'].max():.1f}m")
    print(f"Saved to: {points_file}")
    
    return combined_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ICESat-2 bathymetry data")
    parser.add_argument("--region", required=True, help="Region name (e.g., kachchh)")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        points_df = fetch_icesat2_data(args.region, args.start, args.end)
        
        if points_df is not None:
            print("\n[NEXT STEP] Run the mapping script to align with Sentinel-2 features")
            print(f"python scripts/map_icesat2_to_grid.py --region {args.region}")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch ICESat-2 data: {e}")
        sys.exit(1)