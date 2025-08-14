"""
Test the updated API with real shopping site integration
"""
import requests
import json
import time

# API endpoint
API_URL = "http://localhost:8000"

def test_market_research_api():
    """Test the market research API with real shopping sites"""
    
    # Request data
    data = {
        "product_description": "luxury leather handbag",
        "category": "bags",
        "max_products": 10,
        "use_real_shopping_sites": True
    }
    
    print("Testing market research API with real shopping sites...")
    print(f"Sending request to {API_URL}/research/market")
    
    try:
        # Make the API call
        response = requests.post(f"{API_URL}/research/market", json=data)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Print result summary
            print("\n=== Market Research Results ===")
            print(f"Status: {result['status']}")
            print(f"Total products found: {result['data']['total_products_found']}")
            print(f"Data sources: {result['data']['data_sources']}")
            print(f"Processing time: {result['processing_time']} seconds")
            print("\nTop brands found:")
            for brand in result['data']['top_brands']:
                print(f"  - {brand}")
            
            print("\nSample products:")
            for idx, product in enumerate(result['data']['competitor_products'][:3]):
                print(f"\n{idx+1}. {product['brand']} - {product['name']}")
                print(f"   Price: ${product['price']}")
                print(f"   URL: {product['url']}")
                
            return True
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error testing API: {e}")
        return False

def test_categories_api():
    """Test the categories API endpoint"""
    
    print("\nTesting categories API...")
    
    try:
        response = requests.get(f"{API_URL}/categories")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Categories: {result['data']['categories']}")
            return True
        else:
            print(f"Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error testing categories API: {e}")
        return False

def main():
    """Run all API tests"""
    start_time = time.time()
    
    market_research_success = test_market_research_api()
    categories_success = test_categories_api()
    
    print("\n=== Test Results ===")
    print(f"Market research API: {'✅ Passed' if market_research_success else '❌ Failed'}")
    print(f"Categories API: {'✅ Passed' if categories_success else '❌ Failed'}")
    
    end_time = time.time()
    print(f"\nTotal test time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
