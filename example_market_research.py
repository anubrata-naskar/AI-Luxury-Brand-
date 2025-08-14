"""
Example script showing how to use the MarketResearchAgent with real web scraping
"""
import asyncio
import json
from src.agents.market_agent import MarketResearchAgent, CompetitorProduct

async def main():
    """
    Demonstrate how to use the MarketResearchAgent with real web scraping
    """
    print("Initializing the Market Research Agent...")
    agent = MarketResearchAgent()
    
    # Example query
    product = "luxury cashmere sweater"
    category = "sweaters"
    
    print(f"\nSearching for: {product} in category: {category}")
    print("This will attempt to scrape real product data from luxury shopping sites...\n")
    
    # Search using the agent with real sites
    result = await agent.research_competitors(
        product_description=product,
        category=category,
        max_products=10,
        use_real_shopping_sites=True
    )
    
    # Print the results
    print(f"\n===== MARKET RESEARCH RESULTS =====")
    print(f"Category: {result.category}")
    print(f"Total products found: {result.total_products_found}")
    print(f"Average price: ${result.price_analysis.get('avg_price', 'N/A'):.2f}")
    print(f"Price range: ${result.price_analysis.get('min_price', 'N/A'):.2f} - ${result.price_analysis.get('max_price', 'N/A'):.2f}")
    
    print("\nTop Brands:")
    for brand in result.top_brands[:5]:
        print(f"  - {brand}")
    
    print("\nMarket Trends:")
    for trend in result.market_trends:
        print(f"  - {trend}")
    
    print("\nSample Products:")
    for i, product in enumerate(result.competitor_products[:3]):
        print(f"\n{i+1}. {product.brand} - {product.name}")
        print(f"   Price: ${product.price:.2f}")
        print(f"   URL: {product.url}")
        print(f"   Features: {', '.join(product.features[:3])}")
    
    print("\n===== DATA SOURCES =====")
    print(f"Data was collected from: {', '.join(result.data_sources)}")

if __name__ == "__main__":
    asyncio.run(main())
