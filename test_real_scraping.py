"""
Test script for testing the real web scraping functionality of the MarketResearchAgent
"""
import asyncio
import sys
from loguru import logger
from src.agents.market_agent import MarketResearchAgent

async def test_real_scraping():
    """Test the real scraping functionality"""
    # Initialize the agent
    agent = MarketResearchAgent()
    
    # Test parameters
    product_description = "luxury leather handbag"
    category = "bags"
    max_products = 5  # Small number for testing
    
    logger.info(f"Testing real scraping for: {product_description} in category: {category}")
    
    # Test each shopping site individually
    sites = list(agent.real_shopping_sites.items())
    
    for site_name, site_func in sites:
        logger.info(f"\n\n==== Testing {site_name} ====")
        try:
            products = await site_func(product_description, category, max_products)
            
            logger.info(f"Found {len(products)} products from {site_name}")
            
            # Print first product details
            if products:
                product = products[0]
                logger.info(f"Sample product: {product.brand} - {product.name}")
                logger.info(f"Price: ${product.price}")
                logger.info(f"URL: {product.url}")
                logger.info(f"Image: {product.image_urls[0] if product.image_urls else 'No image'}")
            else:
                logger.warning(f"No products found from {site_name}")
                
        except Exception as e:
            logger.error(f"Error testing {site_name}: {e}")
    
    # Test the full research function with real sites
    logger.info("\n\n==== Testing full research with real sites ====")
    try:
        result = await agent.research_competitors(
            product_description=product_description,
            category=category,
            max_products=10,
            use_real_shopping_sites=True
        )
        
        logger.info(f"Research completed. Found {result.total_products_found} products from {len(result.data_sources)} sources")
        logger.info(f"Data sources: {result.data_sources}")
        
        # Show price analysis
        logger.info(f"Price analysis: {result.price_analysis}")
        
        # Show top brands
        logger.info(f"Top brands: {result.top_brands}")
        
    except Exception as e:
        logger.error(f"Error in full research test: {e}")

if __name__ == "__main__":
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run the test
    asyncio.run(test_real_scraping())
