"""
Quick test script for Copernicus Data Space Ecosystem connection.
"""

from sentinelsat import SentinelAPI
from datetime import datetime, timedelta

def test_connection():
    # Get user credentials
    username = input("Enter your Copernicus username: ")
    password = input("Enter your Copernicus password: ")
    
    try:
        # Connect to the new API endpoint
        print("\nTesting connection...")
        api = SentinelAPI(
            username,
            password,
            'https://catalogue.dataspace.copernicus.eu/odata/v1'
        )
        
        # Try a simple query
        print("Testing search functionality...")
        products = api.query(
            area='POINT(2.3522 48.8566)',  # Paris coordinates
            date=(datetime.now() - timedelta(days=7), datetime.now()),
            platformname='Sentinel-2',
            producttype='S2MSI2A',
            limit=1
        )
        
        if products:
            print("\n✅ Connection successful!")
            print(f"Found {len(products)} products")
            print("\nSample product info:")
            for product_id, product_info in products.items():
                print(f"Title: {product_info['title']}")
                print(f"Date: {product_info['beginposition']}")
                print(f"Size: {product_info['size']}")
                break
        else:
            print("\n✅ Connection successful, but no products found in the last 7 days")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your username and password")
        print("2. Ensure your account is activated")
        print("3. Try again in a few minutes")
        print("4. Check if you're behind a proxy/VPN")
        return False
    
    return True

if __name__ == "__main__":
    test_connection()