#!/usr/bin/env python
"""
Simple script to test the market agent
"""
import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    """Test the market agent functionality"""
    try:
        from src.agents.market_agent import MarketResearchAgent
        print("✅ Successfully imported MarketResearchAgent")
        
        agent = MarketResearchAgent()
        print("✅ Successfully initialized MarketResearchAgent")
        
        # Test with a simple product
        result = await agent.research_competitors(
            product_description="luxury cashmere sweater",
            category="sweaters",
            max_products=3
        )
        
        print(f"✅ Research successful. Found {result.total_products_found} products.")
        print(f"✅ Data sources: {result.data_sources}")
        
        # Test similar products
        similar_products = agent.find_similar_products("black evening dress", max_results=2)
        print(f"✅ Similar products function works. Found {len(similar_products)} products.")
        
        print("\n✅ All market agent tests PASSED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
