#!/usr/bin/env python3
"""
Test script to debug market agent issues
"""
import asyncio
import sys
import os
import traceback

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.agents.market_agent import MarketResearchAgent
    print("✅ Successfully imported MarketResearchAgent")
except ImportError as e:
    print(f"❌ Failed to import MarketResearchAgent: {e}")
    traceback.print_exc()
    sys.exit(1)

async def test_market_agent():
    """Test the market agent functionality"""
    print("\n🧪 Testing Market Research Agent...")
    
    try:
        # Initialize the agent
        print("📡 Initializing market agent...")
        agent = MarketResearchAgent()
        print("✅ Market agent initialized successfully")
        
        # Test with a simple product description
        print("\n🔍 Testing competitor research...")
        
        test_cases = [
            {
                "description": "luxury cashmere sweater",
                "category": "sweaters"
            },
            {
                "description": "designer leather handbag",
                "category": "bags"
            },
            {
                "description": "elegant evening dress",
                "category": "dresses"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['description']} ---")
            
            try:
                result = await agent.research_competitors(
                    product_description=test_case['description'],
                    category=test_case['category'],
                    max_products=10
                )
                
                print(f"✅ Research completed successfully!")
                print(f"📊 Found {result.total_products_found} products")
                print(f"🏢 Data sources: {result.data_sources}")
                print(f"💰 Price range: ${result.price_analysis.get('min_price', 0):.0f} - ${result.price_analysis.get('max_price', 0):.0f}")
                print(f"🏆 Top brands: {', '.join(result.top_brands[:3])}")
                print(f"📈 Sample trends: {', '.join(result.market_trends[:2])}")
                
                # Test specific products
                if result.competitor_products:
                    sample_product = result.competitor_products[0]
                    print(f"📦 Sample product: {sample_product.name} by {sample_product.brand} - ${sample_product.price}")
                else:
                    print("⚠️ No competitor products found")
                
            except Exception as e:
                print(f"❌ Test case {i} failed: {e}")
                traceback.print_exc()
                
        print("\n🎯 Testing similar products functionality...")
        try:
            similar_products = agent.find_similar_products(
                "luxury black evening dress", 
                max_results=5
            )
            print(f"✅ Found {len(similar_products)} similar products")
            for product in similar_products[:3]:
                print(f"  • {product.name} - ${product.price}")
        except Exception as e:
            print(f"❌ Similar products test failed: {e}")
            traceback.print_exc()
            
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Market agent test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting market agent tests...")
    asyncio.run(test_market_agent())
