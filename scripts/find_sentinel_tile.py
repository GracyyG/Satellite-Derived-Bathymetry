"""
Script to determine Sentinel-2 tile ID from coordinates
"""
from shapely.geometry import Point
import geopandas as gpd
import requests
import io
import zipfile

def get_tile_for_coordinates():
    # New Mangalore Port coordinates
    lat, lon = 12.92, 74.82
    point = Point(lon, lat)
    
    # Download Sentinel-2 tiling grid
    print("Downloading Sentinel-2 tiling grid...")
    url = "https://raw.githubusercontent.com/justinelliotmeyers/Sentinel-2-Shapefile-Index/master/sentinel2_tiles_world.geojson"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Read the GeoJSON data
            tiles = gpd.read_file(io.StringIO(response.text))
            
            # Find which tile contains our point
            for idx, tile in tiles.iterrows():
                if tile.geometry.contains(point):
                    print(f"\nFound matching tile:")
                    print(f"Name: {tile['Name']}")
                    print(f"UTM Zone: {tile['UTM_Zone']}")
                    print(f"Latitude Band: {tile['MGRS_LAT']}")
                    print(f"Grid Square: {tile['MGRS_TILE']}")
                    return
                    
            print("\nNo matching tile found for coordinates")
            
        else:
            print("Failed to download tiling grid")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    get_tile_for_coordinates()