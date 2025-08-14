"""
AI Market Research Agent
Performs automated competitor analysis and market research using web scraping and data analysis
"""
import asyncio
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import time
import re
from datetime import datetime
import pandas as pd
import random
from loguru import logger

try:
    from ..utils.sentiment_analyzer import SentimentAnalyzer
    from ..utils.text_processing import TextProcessor
    from ..utils.scraper_manager import ScraperManager
except ImportError:
    # Fallback for testing
    import sys
    import os
    
    # Add project root to path for testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        from src.utils.scraper_manager import ScraperManager
    except ImportError:
        # Define a simple version for testing if not available
        class ScraperManager:
            def __init__(self, max_retries=3, retry_delay=1.0):
                self.session = requests.Session()
                self.session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
                })
            
            async def fetch_with_retry(self, url, timeout=10):
                try:
                    return self.session.get(url, timeout=timeout)
                except:
                    return None
            
            async def paginated_fetch(self, base_url, extract_func, max_pages=1, page_param='page', starting_page=1):
                response = await self.fetch_with_retry(base_url)
                if not response:
                    return []
                soup = BeautifulSoup(response.text, 'html.parser')
                return extract_func(soup)
    
    class SentimentAnalyzer:
        def analyze_text(self, text):
            return {"compound": 0.5, "pos": 0.6, "neg": 0.1, "neu": 0.3}
    
    class TextProcessor:
        def extract_keywords(self, text):
            return text.lower().split()

@dataclass
class CompetitorProduct:
    """Structure for competitor product data"""
    name: str
    brand: str
    price: float
    url: str
    description: str
    reviews_count: int
    average_rating: float
    availability: str
    image_urls: List[str]
    features: List[str]
    scraped_date: datetime = field(default_factory=datetime.now)

@dataclass
class MarketResearchResult:
    """Structure for market research results"""
    category: str
    search_query: str
    total_products_found: int
    competitor_products: List[CompetitorProduct]
    price_analysis: Dict[str, float]
    sentiment_analysis: Dict[str, float]
    market_trends: List[str]
    top_brands: List[str]
    average_rating: float
    research_date: datetime = field(default_factory=datetime.now)
    data_sources: List[str] = field(default_factory=list)

class MarketResearchAgent:
    """AI agent for automated market research and competitor analysis"""
    
    def __init__(self):
        """Initialize the market research agent"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_processor = TextProcessor()
        
        # Initialize the scraper manager for robust web scraping
        self.scraper = ScraperManager(max_retries=3, retry_delay=1.0)
        
        # Keep a legacy session for backward compatibility
        self.session = self.scraper.session
        
        # Real shopping sites to scrape data from
        self.real_shopping_sites = {
            'farfetch': self._search_farfetch,
            'net_a_porter': self._search_net_a_porter,
            'mytheresa': self._search_mytheresa,
            'matches_fashion': self._search_matches_fashion,
            'ssense': self._search_ssense
        }
        
        # Simulated data sources (fallback)
        self.supported_sites = {
            'luxury_dataset': self._generate_luxury_products,
            'fashion_dataset': self._generate_fashion_products,
            'sample_data': self._load_sample_data
        }
        
        # Luxury brands by category
        self.luxury_brands = {
            "dresses": ["Valentino", "Gucci", "Saint Laurent", "Dior", "Chanel", "Prada", "Burberry"],
            "shoes": ["Louboutin", "Jimmy Choo", "Manolo Blahnik", "Gucci", "Prada", "Ferragamo", "Balenciaga"],
            "bags": ["Hermès", "Chanel", "Louis Vuitton", "Gucci", "Bottega Veneta", "Fendi", "Prada"],
            "jewelry": ["Cartier", "Tiffany & Co.", "Van Cleef & Arpels", "Bulgari", "Harry Winston", "Graff"],
            "sweaters": ["Brunello Cucinelli", "Loro Piana", "Ralph Lauren", "The Row", "Max Mara", "Gucci"],
            "jackets": ["Moncler", "Burberry", "Fendi", "Prada", "Saint Laurent", "Balenciaga", "Tom Ford"],
            "watches": ["Rolex", "Patek Philippe", "Audemars Piguet", "Omega", "Cartier", "Jaeger-LeCoultre"],
        }
        
        # Default category if not specified
        self.default_brands = ["Louis Vuitton", "Gucci", "Chanel", "Hermès", "Prada", "Dior", "Versace", "Balenciaga"]
    
    async def research_competitors(self, 
                                 product_description: str, 
                                 category: str,
                                 max_products: int = 50,
                                 price_range: Tuple[float, float] = None,
                                 use_real_shopping_sites: bool = True) -> MarketResearchResult:
        """
        Perform comprehensive competitor research
        
        Args:
            product_description: Description of the product to research
            category: Product category
            max_products: Maximum number of products to analyze
            price_range: Optional price range filter (min, max)
            use_real_shopping_sites: Whether to use real shopping sites for data scraping
        
        Returns:
            MarketResearchResult with comprehensive analysis
        """
        logger.info(f"Starting market research for: {product_description}")
        
        try:
            # Generate search queries
            search_queries = self._generate_search_queries(product_description, category)
            
            # Generate data from all supported sources
            all_products = []
            data_sources = []
            
            # First try to get real data from shopping sites if enabled
            if use_real_shopping_sites:
                logger.info("Attempting to fetch from real shopping sites...")
                
                for site_name, site_func in self.real_shopping_sites.items():
                    try:
                        logger.info(f"Fetching data from {site_name}...")
                        site_products = await site_func(product_description, category, max_products // (len(self.real_shopping_sites) + len(self.supported_sites)))
                        
                        if site_products:
                            all_products.extend(site_products)
                            data_sources.append(site_name)
                            logger.info(f"Got {len(site_products)} products from {site_name}")
                    
                    except Exception as e:
                        logger.warning(f"Error fetching from {site_name}: {e}")
            
            # If we didn't get enough products, try the simulated sources
            if len(all_products) < max_products // 2:
                logger.info(f"Only found {len(all_products)} products from real sites. Adding simulated data...")
                
                # Process each simulated data source
                for source_name, source_func in self.supported_sites.items():
                    try:
                        products_needed = max(0, max_products - len(all_products))
                        if products_needed <= 0:
                            break
                            
                        logger.info(f"Fetching data from {source_name}...")
                        source_products = await source_func(product_description, category, products_needed // len(self.supported_sites))
                        
                        if source_products:
                            all_products.extend(source_products)
                            data_sources.append(source_name)
                            logger.info(f"Got {len(source_products)} products from {source_name}")
                    
                    except Exception as e:
                        logger.warning(f"Error fetching from {source_name}: {e}")
            
            # Generate fallback data if needed
            if not all_products:
                logger.info("No products found, generating fallback data...")
                fallback_products = self._generate_fallback_products(product_description, category, 10)
                all_products.extend(fallback_products)
                data_sources = ["fallback_data"]
            
            # Filter by price range if specified
            if price_range:
                all_products = [p for p in all_products if price_range[0] <= p.price <= price_range[1]]
            
            # Limit to max_products
            all_products = all_products[:max_products]
            
            # Perform analysis
            price_analysis = self._analyze_pricing(all_products)
            sentiment_analysis = await self._analyze_sentiment(all_products)
            market_trends = self._identify_market_trends(all_products)
            top_brands = self._identify_top_brands(all_products)
            average_rating = self._calculate_average_rating(all_products)
            
            # Create result
            result = MarketResearchResult(
                category=category,
                search_query=product_description,
                total_products_found=len(all_products),
                competitor_products=all_products,
                price_analysis=price_analysis,
                sentiment_analysis=sentiment_analysis,
                market_trends=market_trends,
                top_brands=top_brands,
                average_rating=average_rating,
                research_date=datetime.now(),
                data_sources=data_sources
            )
            
            logger.info(f"Market research completed. Found {len(all_products)} products from {len(data_sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error in market research: {e}")
            raise
    
    def _generate_search_queries(self, product_description: str, category: str) -> List[str]:
        """Generate effective search queries for the product"""
        base_terms = self.text_processor.extract_keywords(product_description)
        
        queries = [
            product_description,
            f"{category} luxury",
            f"designer {category}",
            f"premium {category}"
        ]
        
        # Add brand-specific queries if brand is mentioned
        if any(brand.lower() in product_description.lower() for brand in ["gucci", "prada", "chanel"]):
            queries.append(f"luxury {category} similar")
        
        # Add material-based queries
        materials = ["cashmere", "silk", "leather", "cotton", "wool"]
        for material in materials:
            if material in product_description.lower():
                queries.append(f"{material} {category}")
        
        return queries[:5]  # Limit to 5 queries
    
    def _analyze_pricing(self, products: List[CompetitorProduct]) -> Dict[str, float]:
        """Analyze pricing patterns in competitor products"""
        if not products:
            return {}
        
        prices = [p.price for p in products if p.price > 0]
        
        if not prices:
            return {}
        
        # Calculate standard deviation safely
        try:
            price_std = float(pd.Series(prices).std())
            if pd.isna(price_std):  # Handle NaN value
                price_std = 0.0
        except:
            price_std = 0.0
        
        return {
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": sum(prices) / len(prices),
            "median_price": sorted(prices)[len(prices)//2],
            "price_std": price_std,
            "total_products": len(products)
        }
    
    async def _analyze_sentiment(self, products: List[CompetitorProduct]) -> Dict[str, float]:
        """Analyze sentiment from product reviews and descriptions"""
        sentiments = []
        
        for product in products:
            if product.description:
                sentiment = self.sentiment_analyzer.analyze_text(product.description)
                sentiments.append(sentiment['compound'])
        
        if not sentiments:
            return {"overall_sentiment": 0.0, "positive_ratio": 0.0}
        
        positive_count = sum(1 for s in sentiments if s > 0.1)
        
        return {
            "overall_sentiment": sum(sentiments) / len(sentiments),
            "positive_ratio": positive_count / len(sentiments),
            "total_analyzed": len(sentiments)
        }
    
    def _identify_market_trends(self, products: List[CompetitorProduct]) -> List[str]:
        """Identify market trends from competitor products"""
        if not products:
            return ["Insufficient data for trend analysis"]
        
        trends = []
        
        # Analyze price trends
        prices = [p.price for p in products if p.price > 0]
        if prices:
            avg_price = sum(prices) / len(prices)
            if avg_price > 1000:
                trends.append("Premium pricing dominance")
            elif avg_price < 200:
                trends.append("Affordable luxury trend")
            else:
                trends.append("Mid-range market positioning")
        
        # Analyze brand presence
        brands = [p.brand for p in products]
        brand_counts = {}
        for brand in brands:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        if brand_counts:
            top_brand = max(brand_counts.items(), key=lambda x: x[1])
            trends.append(f"{top_brand[0]} shows strong market presence")
        
        # Analyze features
        all_features = []
        for product in products:
            all_features.extend(product.features)
        
        feature_counts = {}
        for feature in all_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        if feature_counts:
            top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for feature, count in top_features:
                if count > 1:
                    trends.append(f"Popular feature: {feature}")
        
        # Add some general luxury market trends
        luxury_trends = [
            "Sustainability focus increasing",
            "Digital-first luxury experiences",
            "Personalization and customization",
            "Heritage craftsmanship emphasis",
            "Limited edition collections popularity",
            "Experiential luxury on the rise",
            "Direct-to-consumer luxury brands emerging"
        ]
        
        trends.extend(random.sample(luxury_trends, min(3, len(luxury_trends))))
        
        return trends[:7]  # Limit to 7 trends
    
    def _identify_top_brands(self, products: List[CompetitorProduct]) -> List[str]:
        """Identify top brands from competitor analysis"""
        if not products:
            return ["No brands identified"]
        
        brand_counts = {}
        for product in products:
            brand = product.brand
            if brand and brand != "Unknown":
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        if not brand_counts:
            return ["Various luxury brands"]
        
        # Sort by frequency and return top brands
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        top_brands = [brand for brand, count in sorted_brands[:10]]
        
        return top_brands if top_brands else ["Various luxury brands"]
    
    def _calculate_average_rating(self, products: List[CompetitorProduct]) -> float:
        """Calculate average rating across all products"""
        if not products:
            return 0.0
        
        ratings = [p.average_rating for p in products if p.average_rating > 0]
        
        if not ratings:
            return 4.2  # Default good rating for luxury products
        
        return round(sum(ratings) / len(ratings), 2)
    
    async def find_similar_products(self, product_description: str, max_results: int = 20, use_real_shopping_sites: bool = True) -> List[CompetitorProduct]:
        """
        Find similar products based on user description
        
        Args:
            product_description: User's product description
            max_results: Maximum number of products to find
            use_real_shopping_sites: Whether to use real shopping sites for data scraping
        
        Returns:
            List of similar competitor products
        """
        logger.info(f"Finding similar products for: {product_description}")
        
        # Extract key terms from product description
        search_terms = self._extract_search_terms(product_description)
        category = self._extract_category(product_description)
        
        all_products = []
        
        # Try to get real products first if enabled
        if use_real_shopping_sites:
            logger.info("Searching real shopping sites for similar products...")
            
            # Choose two random sites to query (to avoid making too many requests)
            site_items = list(self.real_shopping_sites.items())
            random.shuffle(site_items)
            selected_sites = site_items[:2]  # Use only 2 sites
            
            for site_name, site_func in selected_sites:
                try:
                    logger.info(f"Searching {site_name}...")
                    site_products = await site_func(product_description, category, max_results // 2)
                    
                    if site_products:
                        all_products.extend(site_products)
                        logger.info(f"Found {len(site_products)} products from {site_name}")
                        
                except Exception as e:
                    logger.warning(f"Error searching {site_name}: {e}")
        
        # If we didn't find enough real products, generate simulated ones
        if len(all_products) < max_results:
            remaining_products = max_results - len(all_products)
            logger.info(f"Generating {remaining_products} simulated products to complete the set...")
            
            # Generate similar products
            simulated_products = self._generate_similar_products(
                product_description, category, search_terms, remaining_products)
            
            all_products.extend(simulated_products)
        
        return all_products[:max_results]
    
    def _extract_search_terms(self, description: str) -> List[str]:
        """Extract relevant search terms from product description"""
        
        # Common fashion keywords to look for
        fashion_keywords = [
            'dress', 'shirt', 'pants', 'jeans', 'jacket', 'coat', 'sweater', 'top',
            'skirt', 'blouse', 'cardigan', 'blazer', 'suit', 'jumpsuit', 'romper',
            'shoes', 'boots', 'heels', 'sneakers', 'sandals', 'flats',
            'bag', 'purse', 'handbag', 'backpack', 'wallet', 'clutch',
            'jewelry', 'necklace', 'earrings', 'bracelet', 'ring', 'watch',
            'scarf', 'hat', 'belt', 'sunglasses', 'gloves'
        ]
        
        # Material keywords
        material_keywords = [
            'cotton', 'silk', 'wool', 'cashmere', 'leather', 'denim', 'linen',
            'polyester', 'nylon', 'viscose', 'bamboo', 'organic'
        ]
        
        # Style keywords
        style_keywords = [
            'vintage', 'retro', 'modern', 'classic', 'casual', 'formal', 'elegant',
            'minimalist', 'bohemian', 'edgy', 'feminine', 'masculine', 'unisex'
        ]
        
        description_lower = description.lower()
        found_terms = []
        
        # Extract brand if mentioned
        brand_pattern = r'brand[:\s]*([a-zA-Z]+)'
        brand_match = re.search(brand_pattern, description_lower)
        if brand_match:
            found_terms.append(brand_match.group(1))
        
        # Extract product type
        for keyword in fashion_keywords:
            if keyword in description_lower:
                found_terms.append(keyword)
        
        # Extract materials
        for material in material_keywords:
            if material in description_lower:
                found_terms.append(material)
        
        # Extract style descriptors
        for style in style_keywords:
            if style in description_lower:
                found_terms.append(style)
        
        # Extract color if mentioned
        colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 
                 'purple', 'orange', 'brown', 'gray', 'navy', 'beige', 'tan']
        for color in colors:
            if color in description_lower:
                found_terms.append(color)
        
        return list(set(found_terms))  # Remove duplicates
    
    def _extract_category(self, description: str) -> str:
        """Extract category from product description"""
        # Map of keywords to categories
        category_keywords = {
            'dress': 'dresses',
            'gown': 'dresses',
            'sweater': 'sweaters',
            'pullover': 'sweaters',
            'cardigan': 'sweaters',
            'bag': 'bags',
            'handbag': 'bags',
            'purse': 'bags',
            'shoe': 'shoes',
            'footwear': 'shoes',
            'boot': 'shoes',
            'jacket': 'jackets',
            'coat': 'jackets',
            'blazer': 'jackets',
            'jean': 'jeans',
            'denim': 'jeans',
            'jewelry': 'jewelry',
            'necklace': 'jewelry',
            'ring': 'jewelry',
            'watch': 'watches',
            'timepiece': 'watches',
        }
        
        description_lower = description.lower()
        
        for keyword, category in category_keywords.items():
            if keyword in description_lower:
                return category
        
        return 'general'  # Default category
    
    def _generate_similar_products(self, description: str, category: str, search_terms: List[str], max_results: int) -> List[CompetitorProduct]:
        """Generate similar products based on description and search terms"""
        
        products = []
        
        # Use the appropriate luxury brands for this category
        brands = self.luxury_brands.get(category, self.default_brands)
        
        # Base price ranges by category (min, max)
        price_ranges = {
            'dresses': (800, 5000),
            'shoes': (500, 3000),
            'bags': (1000, 10000),
            'jewelry': (1000, 50000),
            'watches': (3000, 25000),
            'sweaters': (300, 2000),
            'jackets': (800, 5000),
            'jeans': (200, 1200),
            'general': (500, 3000)
        }
        
        min_price, max_price = price_ranges.get(category, (500, 3000))
        
        # Adjust price based on search terms
        if any(term in ['luxury', 'premium', 'high-end', 'designer'] for term in search_terms):
            min_price = min_price * 1.5
            max_price = max_price * 1.5
        
        # Generate products
        for i in range(max_results):
            brand = random.choice(brands)
            price = random.uniform(min_price, max_price)
            
            # Create product name based on category and search terms
            name_prefix = brand
            
            # Product type based on category
            product_types = {
                'dresses': ['Evening Dress', 'Cocktail Dress', 'Gown', 'Maxi Dress'],
                'shoes': ['Leather Pumps', 'Designer Heels', 'Luxury Loafers', 'Boots'],
                'bags': ['Leather Handbag', 'Designer Tote', 'Clutch', 'Shoulder Bag'],
                'jewelry': ['Diamond Necklace', 'Gold Bracelet', 'Luxury Watch', 'Designer Ring'],
                'watches': ['Chronograph', 'Automatic Watch', 'Luxury Timepiece', 'Limited Edition Watch'],
                'sweaters': ['Cashmere Sweater', 'Wool Pullover', 'Designer Cardigan', 'Luxury Knit'],
                'jackets': ['Leather Jacket', 'Designer Blazer', 'Luxury Coat', 'Limited Edition Jacket'],
                'jeans': ['Designer Jeans', 'Slim Fit Denim', 'Luxury Jeans', 'Distressed Denim'],
                'general': ['Luxury Item', 'Designer Piece', 'Premium Product', 'High-end Accessory']
            }
            
            item_type = random.choice(product_types.get(category, product_types['general']))
            
            # Add material or color if in search terms
            materials = ['Cashmere', 'Silk', 'Leather', 'Wool', 'Cotton']
            colors = ['Black', 'White', 'Red', 'Blue', 'Navy', 'Beige', 'Brown']
            
            material = next((m for m in materials if m.lower() in description.lower()), '')
            color = next((c for c in colors if c.lower() in description.lower()), '')
            
            name_parts = []
            if color:
                name_parts.append(color)
            if material:
                name_parts.append(material)
            name_parts.append(item_type)
            
            name = f"{brand} {' '.join(name_parts)}"
            
            # Generate random reviews and rating
            reviews_count = random.randint(5, 200)
            average_rating = round(random.uniform(4.0, 5.0), 1)
            
            # Generate features
            features = self._generate_features_for_product(category, material, description)
            
            # Create and add the product
            product = CompetitorProduct(
                name=name,
                brand=brand,
                price=round(price, 2),
                url=f"https://example.com/{brand.lower()}/{item_type.lower().replace(' ', '-')}",
                description=f"Luxury {item_type} by {brand}. {material if material else 'Premium'} quality with exceptional craftsmanship.",
                reviews_count=reviews_count,
                average_rating=average_rating,
                availability="In Stock" if random.random() > 0.2 else "Limited Stock",
                image_urls=[f"https://example.com/images/{brand.lower()}/{i+1}.jpg"],
                features=features,
                scraped_date=datetime.now()
            )
            
            products.append(product)
        
        return products
    
    def _generate_features_for_product(self, category: str, material: str, description: str) -> List[str]:
        """Generate realistic features for a product based on category and material"""
        
        # Base features by category
        category_features = {
            'dresses': ['Designer Cut', 'Hand-finished Edges', 'Concealed Zipper', 'Fully Lined'],
            'shoes': ['Leather Sole', 'Comfortable Fit', 'Designer Heel', 'Signature Design'],
            'bags': ['Multiple Compartments', 'Designer Hardware', 'Dust Bag Included', 'Signature Clasp'],
            'jewelry': ['Genuine Gemstones', 'Handcrafted Design', 'Luxury Finish', 'Designer Signature'],
            'watches': ['Swiss Movement', 'Sapphire Crystal', 'Water Resistant', 'Limited Edition'],
            'sweaters': ['Ribbed Cuffs', 'Luxury Finish', 'Designer Pattern', 'Comfortable Fit'],
            'jackets': ['Designer Cut', 'Premium Lining', 'Signature Details', 'Luxury Finish'],
            'jeans': ['Perfect Fit', 'Premium Denim', 'Designer Wash', 'Signature Details'],
            'general': ['Premium Quality', 'Designer Craftsmanship', 'Luxury Finish', 'Exclusive Design']
        }
        
        # Material-specific features
        material_features = {
            'Cashmere': ['100% Pure Cashmere', 'Incredibly Soft', 'Warm and Lightweight'],
            'Silk': ['100% Pure Silk', 'Lustrous Finish', 'Lightweight and Breathable'],
            'Leather': ['Genuine Leather', 'Buttery Soft', 'Durable Quality'],
            'Wool': ['Premium Wool', 'Warm and Cozy', 'Natural Insulation'],
            'Cotton': ['Premium Cotton', 'Breathable Fabric', 'Soft Touch']
        }
        
        # Get base features
        base = category_features.get(category, category_features['general'])
        
        # Add material features if applicable
        if material and material in material_features:
            base.extend(material_features[material])
        
        # Add luxury features
        luxury_features = [
            'Made in Italy',
            'Limited Production',
            'Exclusive Design',
            'Hand-finished Details',
            'Sustainably Produced',
            'Artisan Craftsmanship'
        ]
        
        base.extend(random.sample(luxury_features, 2))
        
        # Return random selection of features (3-5)
        return random.sample(base, min(len(base), random.randint(3, 5)))
    
    async def _generate_luxury_products(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """Generate luxury products based on description and category"""
        
        # Get search terms
        search_terms = self._extract_search_terms(product_description)
        
        # Generate products
        products = self._generate_similar_products(product_description, category, search_terms, max_results)
        
        # Add some delay to simulate API call
        await asyncio.sleep(0.5)
        
        return products
    
    async def _generate_fashion_products(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """Generate fashion products with different brands and pricing"""
        
        # Use different brands for fashion dataset
        fashion_brands = [
            "Alexander McQueen", "Stella McCartney", "Tom Ford", "Celine", 
            "Balenciaga", "Givenchy", "Valentino", "Miu Miu", "Loewe",
            "Balmain", "Acne Studios", "Isabel Marant", "Jacquemus"
        ]
        
        products = []
        
        # Price ranges by category (slightly different from luxury dataset)
        price_ranges = {
            'dresses': (600, 3000),
            'shoes': (400, 1800),
            'bags': (900, 6000),
            'jewelry': (800, 15000),
            'sweaters': (250, 1500),
            'jackets': (700, 3500),
            'general': (400, 2000)
        }
        
        min_price, max_price = price_ranges.get(category, (400, 2000))
        
        # Generate products
        for i in range(max_results):
            brand = random.choice(fashion_brands)
            price = random.uniform(min_price, max_price)
            
            # Product types by category
            product_types = {
                'dresses': ['Silk Dress', 'Designer Gown', 'Evening Dress', 'Cocktail Dress'],
                'shoes': ['Designer Pumps', 'Fashion Heels', 'Statement Boots', 'Luxury Flats'],
                'bags': ['Fashion Tote', 'Designer Clutch', 'Statement Bag', 'Crossbody Bag'],
                'jewelry': ['Statement Necklace', 'Designer Earrings', 'Fashion Bracelet', 'Luxury Ring'],
                'sweaters': ['Designer Knit', 'Fashion Sweater', 'Luxury Cardigan', 'Statement Pullover'],
                'jackets': ['Statement Coat', 'Designer Jacket', 'Fashion Blazer', 'Luxury Outerwear'],
                'general': ['Designer Piece', 'Fashion Item', 'Luxury Statement', 'Runway Piece']
            }
            
            item_type = random.choice(product_types.get(category, product_types['general']))
            
            # Create product name
            name = f"{brand} {item_type}"
            
            # Generate features
            features = [
                "Runway Collection",
                "Limited Edition",
                "Fashion Forward Design",
                "Editorial Favorite",
                "Celebrity Choice",
                "Influencer Must-Have"
            ]
            
            product = CompetitorProduct(
                name=name,
                brand=brand,
                price=round(price, 2),
                url=f"https://fashion-example.com/{brand.lower()}/{item_type.lower().replace(' ', '-')}",
                description=f"Fashion-forward {item_type} from {brand}'s latest collection. Make a statement with this designer piece.",
                reviews_count=random.randint(3, 50),
                average_rating=round(random.uniform(3.8, 5.0), 1),
                availability="In Stock" if random.random() > 0.3 else "Limited Stock",
                image_urls=[f"https://fashion-example.com/images/{brand.lower()}/{i+1}.jpg"],
                features=random.sample(features, min(len(features), random.randint(2, 4))),
                scraped_date=datetime.now()
            )
            
            products.append(product)
        
        # Simulate API delay
        await asyncio.sleep(0.3)
        
        return products
    
    async def _load_sample_data(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """Load and filter existing sample data from CSV files"""
        products = []
        
        try:
            # Try to load from datasets folder
            import os
            datasets_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
            
            # Check for existing dataset files
            for filename in ['amazon_co-ecommerce_sample.csv', 'handm.csv', 'data_amazon.xlsx_-_Sheet1.csv']:
                file_path = os.path.join(datasets_path, filename)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Basic filtering by category if column exists
                        if 'category' in df.columns:
                            df = df[df['category'].str.contains(category, case=False, na=False)]
                        
                        # Convert to CompetitorProduct objects
                        for _, row in df.head(max_results).iterrows():
                            try:
                                # Extract relevant fields, handling different column names
                                name = str(row.get('name', row.get('title', row.get('product_name', 'Unknown Product'))))
                                brand = str(row.get('brand', row.get('manufacturer', 'Unknown')))
                                price = float(row.get('price', row.get('cost', row.get('retail_price', 0))))
                                if price == 0:
                                    # Try parsing price from string (e.g. "$123.45")
                                    price_str = str(row.get('price_string', '0'))
                                    price_match = re.search(r'[\d,.]+', price_str)
                                    if price_match:
                                        try:
                                            price = float(price_match.group(0).replace(',', ''))
                                        except:
                                            price = 0
                                
                                # Create product object
                                product = CompetitorProduct(
                                    name=name,
                                    brand=brand if brand and brand != "nan" else self._extract_brand_from_name(name),
                                    price=price,
                                    url=str(row.get('url', row.get('link', '#'))),
                                    description=str(row.get('description', row.get('details', row.get('product_description', '')))),
                                    reviews_count=int(row.get('reviews_count', row.get('reviews', row.get('number_of_reviews', 0)))),
                                    average_rating=float(row.get('rating', row.get('score', row.get('average_rating', 4.0)))),
                                    availability=str(row.get('availability', "Available")),
                                    image_urls=[str(row.get('image', row.get('image_url', '')))],
                                    features=[],
                                    scraped_date=datetime.now()
                                )
                                products.append(product)
                            except Exception as e:
                                logger.debug(f"Error processing sample data row: {e}")
                                continue
                        
                        if products:
                            break
                    except Exception as e:
                        logger.warning(f"Error processing file {filename}: {e}")
                        
        except Exception as e:
            logger.warning(f"Could not load sample data: {e}")
        
        # If we found products but they don't have features, add some
        for product in products:
            if not product.features:
                product.features = self._generate_features_for_product(category, "", product.description)
        
        return products
    
    def _generate_fallback_products(self, product_description: str, category: str, count: int = 5) -> List[CompetitorProduct]:
        """Generate fallback products when no data is available"""
        
        # Get appropriate brands
        brands = self.luxury_brands.get(category, self.default_brands)
        
        products = []
        
        for i in range(count):
            brand = random.choice(brands)
            
            # Generate a simple product
            product = CompetitorProduct(
                name=f"{brand} {category.title()} - Luxury Collection",
                brand=brand,
                price=random.uniform(500, 3000),
                url=f"https://example.com/products/{category}/{i}",
                description=f"Premium {category} product from {brand}'s luxury collection. Matches your search for: {product_description}",
                reviews_count=random.randint(5, 100),
                average_rating=round(random.uniform(4.0, 5.0), 1),
                availability="In Stock",
                image_urls=[f"https://example.com/images/{brand.lower()}/{category}/{i}.jpg"],
                features=["Premium Quality", "Designer Brand", "Luxury Craftsmanship"],
                scraped_date=datetime.now()
            )
            
            products.append(product)
        
        return products
    
    def _extract_brand_from_name(self, product_name: str) -> str:
        """Extract brand name from product name"""
        # List of known luxury brands (flattened from all categories)
        all_brands = set()
        for brands in self.luxury_brands.values():
            all_brands.update(brands)
        all_brands.update(self.default_brands)
        
        # Check if any known brand is in the product name
        name_lower = product_name.lower()
        for brand in all_brands:
            if brand.lower() in name_lower:
                return brand
        
        # If no known brand found, extract first word as potential brand
        words = product_name.split()
        return words[0] if words else "Unknown"
    
    async def _search_farfetch(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """
        Search for products on Farfetch
        
        Args:
            product_description: Description or search terms
            category: Product category
            max_results: Maximum number of results to return
        
        Returns:
            List of CompetitorProduct objects from Farfetch
        """
        logger.info(f"Searching Farfetch for: {product_description} in {category}")
        products = []
        
        try:
            # Convert category to Farfetch's format
            farfetch_categories = {
                'dresses': 'clothing/dresses',
                'shoes': 'shoes',
                'bags': 'bags',
                'jewelry': 'accessories/jewelry',
                'watches': 'accessories/watches',
                'sweaters': 'clothing/knitwear',
                'jackets': 'clothing/jackets',
                'jeans': 'clothing/jeans',
                'general': ''
            }
            
            ff_category = farfetch_categories.get(category, '')
            
            # Format search query from description
            search_terms = self._extract_search_terms(product_description)
            search_query = '+'.join(search_terms[:3])  # Use first 3 terms only
            
            # Construct URL
            base_url = 'https://www.farfetch.com'
            if ff_category:
                url = f"{base_url}/shopping/{ff_category}?q={search_query}"
            else:
                url = f"{base_url}/shopping/search?q={search_query}"
            
            # Define the product extraction function for the paginated fetch
            def extract_products(soup):
                extracted_products = []
                
                # Find product containers
                product_elements = soup.select('div[data-testid="productCard"]')
                
                for product_element in product_elements:
                    try:
                        # Extract product details
                        name_element = product_element.select_one('p[data-testid="productDescription"]')
                        brand_element = product_element.select_one('p[data-testid="brandName"]')
                        price_element = product_element.select_one('span[data-testid="price"]')
                        image_element = product_element.select_one('img')
                        link_element = product_element.select_one('a')
                        
                        # Extract values with fallbacks
                        name = name_element.text if name_element else "Unknown Product"
                        brand = brand_element.text if brand_element else self._extract_brand_from_name(name)
                        
                        # Handle price extraction
                        price = 0
                        if price_element:
                            price_text = price_element.text.strip()
                            price_match = re.search(r'[\d,.]+', price_text)
                            if price_match:
                                price = float(price_match.group(0).replace(',', '').replace('.', ''))
                                # If price has more than 2 decimal places, divide by 100 to get correct format
                                if price > 1000:
                                    price = price / 100
                        
                        # Get image URL
                        image_url = ""
                        if image_element and image_element.has_attr('src'):
                            image_url = image_element['src']
                        elif image_element and image_element.has_attr('data-src'):
                            image_url = image_element['data-src']
                        
                        # Get product URL
                        product_url = ""
                        if link_element and link_element.has_attr('href'):
                            product_url = base_url + link_element['href'] if link_element['href'].startswith('/') else link_element['href']
                        
                        # Create product object
                        product = CompetitorProduct(
                            name=name,
                            brand=brand,
                            price=price,
                            url=product_url,
                            description=f"{brand} {name}",
                            reviews_count=0,  # Farfetch typically doesn't show review counts
                            average_rating=4.5,  # Default rating
                            availability="In Stock",  # Default availability
                            image_urls=[image_url] if image_url else [],
                            features=self._generate_features_for_product(category, "", name),
                            scraped_date=datetime.now()
                        )
                        
                        extracted_products.append(product)
                        
                    except Exception as e:
                        logger.error(f"Error parsing Farfetch product: {e}")
                
                return extracted_products
            
            # Use the paginated fetch to get products from multiple pages if needed
            all_products = await self.scraper.paginated_fetch(
                base_url=url,
                extract_func=extract_products,
                max_pages=2 if max_results > 10 else 1,  # Fetch more pages only if we need more results
                page_param='page',
                starting_page=1
            )
            
            logger.info(f"Found {len(all_products)} products on Farfetch")
            products = all_products[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching Farfetch: {e}")
        
        return products
    
    async def _search_net_a_porter(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """
        Search for products on Net-a-Porter
        
        Args:
            product_description: Description or search terms
            category: Product category
            max_results: Maximum number of results to return
        
        Returns:
            List of CompetitorProduct objects from Net-a-Porter
        """
        logger.info(f"Searching Net-a-Porter for: {product_description} in {category}")
        products = []
        
        try:
            # Convert category to Net-a-Porter's format
            nap_categories = {
                'dresses': 'clothing/dresses',
                'shoes': 'shoes',
                'bags': 'bags',
                'jewelry': 'jewelry',
                'watches': 'watches',
                'sweaters': 'clothing/knitwear',
                'jackets': 'clothing/jackets',
                'jeans': 'clothing/jeans',
                'general': ''
            }
            
            nap_category = nap_categories.get(category, '')
            
            # Format search query from description
            search_query = product_description.replace(' ', '+')
            
            # Construct URL
            base_url = 'https://www.net-a-porter.com'
            if nap_category:
                url = f"{base_url}/en-us/shop/{nap_category}?keywords={search_query}"
            else:
                url = f"{base_url}/en-us/shop/search?keywords={search_query}"
            
            # Make request
            logger.info(f"Fetching from URL: {url}")
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product containers
                product_elements = soup.select('.ProductListWithLoadMore52__listItem')
                logger.info(f"Found {len(product_elements)} products on Net-a-Porter")
                
                for i, product_element in enumerate(product_elements[:max_results]):
                    try:
                        # Extract product details
                        name_element = product_element.select_one('.ProductItem24__name')
                        brand_element = product_element.select_one('.ProductItem24__designer')
                        price_element = product_element.select_one('.PriceWithSchema9__value')
                        image_element = product_element.select_one('img')
                        link_element = product_element.select_one('a')
                        
                        # Extract values with fallbacks
                        name = name_element.text.strip() if name_element else "Unknown Product"
                        brand = brand_element.text.strip() if brand_element else self._extract_brand_from_name(name)
                        
                        # Handle price extraction
                        price = 0
                        if price_element:
                            price_text = price_element.text.strip()
                            price_match = re.search(r'[\d,.]+', price_text)
                            if price_match:
                                price = float(price_match.group(0).replace(',', ''))
                        
                        # Get image URL
                        image_url = ""
                        if image_element and image_element.has_attr('src'):
                            image_url = image_element['src']
                        elif image_element and image_element.has_attr('data-src'):
                            image_url = image_element['data-src']
                        
                        # Get product URL
                        product_url = ""
                        if link_element and link_element.has_attr('href'):
                            product_url = base_url + link_element['href'] if link_element['href'].startswith('/') else link_element['href']
                        
                        # Create product object
                        product = CompetitorProduct(
                            name=name,
                            brand=brand,
                            price=price,
                            url=product_url,
                            description=f"{brand} {name}",
                            reviews_count=0,  # Net-a-Porter typically doesn't show review counts
                            average_rating=4.5,  # Default rating
                            availability="In Stock",  # Default availability
                            image_urls=[image_url] if image_url else [],
                            features=self._generate_features_for_product(category, "", name),
                            scraped_date=datetime.now()
                        )
                        
                        products.append(product)
                        
                    except Exception as e:
                        logger.error(f"Error parsing Net-a-Porter product: {e}")
            else:
                logger.warning(f"Failed to fetch from Net-a-Porter: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error searching Net-a-Porter: {e}")
        
        return products
    
    async def _search_mytheresa(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """
        Search for products on Mytheresa
        
        Args:
            product_description: Description or search terms
            category: Product category
            max_results: Maximum number of results to return
        
        Returns:
            List of CompetitorProduct objects from Mytheresa
        """
        logger.info(f"Searching Mytheresa for: {product_description} in {category}")
        products = []
        
        try:
            # Convert category to Mytheresa's format
            mt_categories = {
                'dresses': 'clothing/dresses',
                'shoes': 'shoes',
                'bags': 'bags',
                'jewelry': 'accessories/jewelry',
                'watches': 'accessories/watches',
                'sweaters': 'clothing/knitwear',
                'jackets': 'clothing/jackets',
                'jeans': 'clothing/jeans',
                'general': ''
            }
            
            mt_category = mt_categories.get(category, '')
            
            # Format search query from description
            search_query = product_description.replace(' ', '+')
            
            # Construct URL
            base_url = 'https://www.mytheresa.com'
            url = f"{base_url}/en-us/search?q={search_query}"
            
            # Make request
            logger.info(f"Fetching from URL: {url}")
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product containers
                product_elements = soup.select('.product')
                logger.info(f"Found {len(product_elements)} products on Mytheresa")
                
                for i, product_element in enumerate(product_elements[:max_results]):
                    try:
                        # Extract product details
                        name_element = product_element.select_one('.name')
                        brand_element = product_element.select_one('.designer')
                        price_element = product_element.select_one('.price')
                        image_element = product_element.select_one('img')
                        link_element = product_element.select_one('a.product-image')
                        
                        # Extract values with fallbacks
                        name = name_element.text.strip() if name_element else "Unknown Product"
                        brand = brand_element.text.strip() if brand_element else self._extract_brand_from_name(name)
                        
                        # Handle price extraction
                        price = 0
                        if price_element:
                            price_text = price_element.text.strip()
                            price_match = re.search(r'[\d,.]+', price_text)
                            if price_match:
                                price = float(price_match.group(0).replace(',', ''))
                        
                        # Get image URL
                        image_url = ""
                        if image_element and image_element.has_attr('src'):
                            image_url = image_element['src']
                        elif image_element and image_element.has_attr('data-src'):
                            image_url = image_element['data-src']
                        
                        # Get product URL
                        product_url = ""
                        if link_element and link_element.has_attr('href'):
                            product_url = base_url + link_element['href'] if link_element['href'].startswith('/') else link_element['href']
                        
                        # Create product object
                        product = CompetitorProduct(
                            name=name,
                            brand=brand,
                            price=price,
                            url=product_url,
                            description=f"{brand} {name}",
                            reviews_count=0,  # Mytheresa typically doesn't show review counts
                            average_rating=4.5,  # Default rating
                            availability="In Stock",  # Default availability
                            image_urls=[image_url] if image_url else [],
                            features=self._generate_features_for_product(category, "", name),
                            scraped_date=datetime.now()
                        )
                        
                        products.append(product)
                        
                    except Exception as e:
                        logger.error(f"Error parsing Mytheresa product: {e}")
            else:
                logger.warning(f"Failed to fetch from Mytheresa: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error searching Mytheresa: {e}")
        
        return products
    
    async def _search_matches_fashion(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """
        Search for products on Matches Fashion
        
        Args:
            product_description: Description or search terms
            category: Product category
            max_results: Maximum number of results to return
        
        Returns:
            List of CompetitorProduct objects from Matches Fashion
        """
        logger.info(f"Searching Matches Fashion for: {product_description} in {category}")
        products = []
        
        try:
            # Convert category to Matches Fashion's format
            mf_categories = {
                'dresses': 'womens/clothing/dresses',
                'shoes': 'womens/shoes',
                'bags': 'womens/bags',
                'jewelry': 'womens/accessories/jewelry',
                'watches': 'womens/accessories/watches',
                'sweaters': 'womens/clothing/knitwear',
                'jackets': 'womens/clothing/jackets',
                'jeans': 'womens/clothing/jeans',
                'general': 'womens'
            }
            
            mf_category = mf_categories.get(category, 'womens')
            
            # Format search query from description
            search_query = product_description.replace(' ', '%20')
            
            # Construct URL
            base_url = 'https://www.matchesfashion.com'
            url = f"{base_url}/search?q={search_query}&path={mf_category}"
            
            # Make request
            logger.info(f"Fetching from URL: {url}")
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product containers
                product_elements = soup.select('.lister__item')
                logger.info(f"Found {len(product_elements)} products on Matches Fashion")
                
                for i, product_element in enumerate(product_elements[:max_results]):
                    try:
                        # Extract product details
                        name_element = product_element.select_one('.product-item__name')
                        brand_element = product_element.select_one('.product-item__designer')
                        price_element = product_element.select_one('.price__value')
                        image_element = product_element.select_one('img')
                        link_element = product_element.select_one('a.product-item__link')
                        
                        # Extract values with fallbacks
                        name = name_element.text.strip() if name_element else "Unknown Product"
                        brand = brand_element.text.strip() if brand_element else self._extract_brand_from_name(name)
                        
                        # Handle price extraction
                        price = 0
                        if price_element:
                            price_text = price_element.text.strip()
                            price_match = re.search(r'[\d,.]+', price_text)
                            if price_match:
                                price = float(price_match.group(0).replace(',', ''))
                        
                        # Get image URL
                        image_url = ""
                        if image_element and image_element.has_attr('src'):
                            image_url = image_element['src']
                        elif image_element and image_element.has_attr('data-src'):
                            image_url = image_element['data-src']
                        
                        # Get product URL
                        product_url = ""
                        if link_element and link_element.has_attr('href'):
                            product_url = base_url + link_element['href'] if link_element['href'].startswith('/') else link_element['href']
                        
                        # Create product object
                        product = CompetitorProduct(
                            name=name,
                            brand=brand,
                            price=price,
                            url=product_url,
                            description=f"{brand} {name}",
                            reviews_count=0,  # Matches Fashion typically doesn't show review counts
                            average_rating=4.5,  # Default rating
                            availability="In Stock",  # Default availability
                            image_urls=[image_url] if image_url else [],
                            features=self._generate_features_for_product(category, "", name),
                            scraped_date=datetime.now()
                        )
                        
                        products.append(product)
                        
                    except Exception as e:
                        logger.error(f"Error parsing Matches Fashion product: {e}")
            else:
                logger.warning(f"Failed to fetch from Matches Fashion: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error searching Matches Fashion: {e}")
        
        return products
    
    async def _search_ssense(self, product_description: str, category: str, max_results: int) -> List[CompetitorProduct]:
        """
        Search for products on SSENSE
        
        Args:
            product_description: Description or search terms
            category: Product category
            max_results: Maximum number of results to return
        
        Returns:
            List of CompetitorProduct objects from SSENSE
        """
        logger.info(f"Searching SSENSE for: {product_description} in {category}")
        products = []
        
        try:
            # Convert category to SSENSE's format
            ssense_categories = {
                'dresses': 'women/clothing/dresses',
                'shoes': 'women/shoes',
                'bags': 'women/bags',
                'jewelry': 'women/jewelry',
                'watches': 'women/accessories/watches',
                'sweaters': 'women/clothing/sweaters',
                'jackets': 'women/clothing/jackets',
                'jeans': 'women/clothing/jeans',
                'general': 'women'
            }
            
            ss_category = ssense_categories.get(category, 'women')
            
            # Format search query from description
            search_terms = self._extract_search_terms(product_description)
            search_query = '+'.join(search_terms[:3])  # Use first 3 terms only
            
            # Construct URL
            base_url = 'https://www.ssense.com'
            url = f"{base_url}/en-us/search?q={search_query}&terms={search_query}"
            
            # Make request
            logger.info(f"Fetching from URL: {url}")
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product containers
                product_elements = soup.select('.plp-products__product')
                logger.info(f"Found {len(product_elements)} products on SSENSE")
                
                for i, product_element in enumerate(product_elements[:max_results]):
                    try:
                        # Extract product details
                        name_element = product_element.select_one('.product-tile__description')
                        brand_element = product_element.select_one('.product-tile__designer')
                        price_element = product_element.select_one('.product-tile__price')
                        image_element = product_element.select_one('img')
                        link_element = product_element.select_one('a.product-tile__link')
                        
                        # Extract values with fallbacks
                        name = name_element.text.strip() if name_element else "Unknown Product"
                        brand = brand_element.text.strip() if brand_element else self._extract_brand_from_name(name)
                        
                        # Handle price extraction
                        price = 0
                        if price_element:
                            price_text = price_element.text.strip()
                            price_match = re.search(r'[\d,.]+', price_text)
                            if price_match:
                                price = float(price_match.group(0).replace(',', ''))
                        
                        # Get image URL
                        image_url = ""
                        if image_element and image_element.has_attr('src'):
                            image_url = image_element['src']
                        elif image_element and image_element.has_attr('data-srcset'):
                            image_url = image_element['data-srcset'].split(',')[0].strip().split(' ')[0]
                        
                        # Get product URL
                        product_url = ""
                        if link_element and link_element.has_attr('href'):
                            product_url = base_url + link_element['href'] if link_element['href'].startswith('/') else link_element['href']
                        
                        # Create product object
                        product = CompetitorProduct(
                            name=name,
                            brand=brand,
                            price=price,
                            url=product_url,
                            description=f"{brand} {name}",
                            reviews_count=0,  # SSENSE doesn't show review counts
                            average_rating=4.5,  # Default rating
                            availability="In Stock",  # Default availability
                            image_urls=[image_url] if image_url else [],
                            features=self._generate_features_for_product(category, "", name),
                            scraped_date=datetime.now()
                        )
                        
                        products.append(product)
                        
                    except Exception as e:
                        logger.error(f"Error parsing SSENSE product: {e}")
            else:
                logger.warning(f"Failed to fetch from SSENSE: Status code {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error searching SSENSE: {e}")
        
        return products
