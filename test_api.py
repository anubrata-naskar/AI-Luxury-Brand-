#!/usr/bin/env python
"""
Test API client for market research
"""
import requests
import json

def test_api():
    """Test the market research API endpoint"""
    print("Testing market research API...")
    
    url = "http://localhost:8000/research/market"
    
    data = {
        "product_description": "luxury cashmere sweater",
        "category": "sweaters",
        "max_products": 5,
        "min_price": 100,
        "max_price": 5000
    }
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            print(f"Status: {result.get('status')}")
            print(f"Processing time: {result.get('processing_time'):.2f} seconds")
            
            data = result.get('data', {})
            print(f"Found {data.get('total_products_found')} products")
            print(f"Top brands: {', '.join(data.get('top_brands', [])[:3])}")
            
            if data.get('competitor_products'):
                sample = data['competitor_products'][0]
                print(f"Sample product: {sample.get('name')} by {sample.get('brand')} - ${sample.get('price')}")
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")

if __name__ == "__main__":
    test_api()
