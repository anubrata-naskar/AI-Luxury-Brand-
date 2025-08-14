"""
AI Market Research Agent
Performs automated competitor analysis and market research using web scraping
"""
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import time
import re
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from ..utils.sentiment_analyzer import SentimentAnalyzer
from ..utils.text_processing import TextProcessor

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
    scraped_date: datetime

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
    research_date: datetime
    data_sources: List[str]

class MarketResearchAgent:
    """AI agent for automated market research and competitor analysis"""
    
    def __init__(self):
        """Initialize the market research agent"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_processor = TextProcessor()
        self.session = requests.Session()
        
        # Configure session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Supported e-commerce sites
        self.supported_sites = {
            'amazon': self._scrape_amazon,
            'nordstrom': self._scrape_nordstrom,
            'saks': self._scrape_saks,
            'net_a_porter': self._scrape_net_a_porter,
            'farfetch': self._scrape_farfetch
        }
    
    async def research_competitors(self, 
                                 product_description: str, 
                                 category: str,
                                 max_products: int = 50,
                                 price_range: Tuple[float, float] = None) -> MarketResearchResult:
        """
        Perform comprehensive competitor research
        
        Args:
            product_description: Description of the product to research
            category: Product category
            max_products: Maximum number of products to analyze
            price_range: Optional price range filter (min, max)
        
        Returns:
            MarketResearchResult with comprehensive analysis
        """
        logger.info(f"Starting market research for: {product_description}")
        
        try:
            # Generate search queries
            search_queries = self._generate_search_queries(product_description, category)
            
            # Scrape multiple sources
            all_products = []
            data_sources = []
            
            for site_name, scraper_func in self.supported_sites.items():
                try:
                    logger.info(f"Scraping {site_name}...")
                    products = await self._scrape_site_safely(
                        site_name, scraper_func, search_queries, max_products // len(self.supported_sites)
                    )
                    all_products.extend(products)
                    data_sources.append(site_name)
                    
                    # Rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape {site_name}: {e}")
                    continue
            
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
    
    async def _scrape_site_safely(self, site_name: str, scraper_func, queries: List[str], max_per_site: int):
        """Safely scrape a site with error handling and rate limiting"""
        products = []
        
        for query in queries:
            try:
                site_products = await scraper_func(query, max_per_site // len(queries))
                products.extend(site_products)
                
                # Rate limiting between queries
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Failed to scrape {site_name} with query '{query}': {e}")
                continue
        
        return products
    
    async def _scrape_amazon(self, query: str, max_results: int) -> List[CompetitorProduct]:
        """Scrape Amazon for competitor products"""
        products = []
        
        try:
            # Setup Selenium for dynamic content
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # Search URL
            search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}&ref=sr_pg_1"
            driver.get(search_url)
            
            # Wait for results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-component-type='s-search-result']"))
            )
            
            # Extract product information
            product_elements = driver.find_elements(By.CSS_SELECTOR, "[data-component-type='s-search-result']")
            
            for element in product_elements[:max_results]:
                try:
                    # Extract product details
                    name_elem = element.find_element(By.CSS_SELECTOR, "h2 a span")
                    name = name_elem.text if name_elem else "Unknown"
                    
                    price_elem = element.find_element(By.CSS_SELECTOR, ".a-price-whole")
                    price_text = price_elem.text if price_elem else "0"
                    price = float(re.sub(r'[^\d.]', '', price_text)) if price_text else 0.0
                    
                    url_elem = element.find_element(By.CSS_SELECTOR, "h2 a")
                    url = "https://amazon.com" + url_elem.get_attribute("href") if url_elem else ""
                    
                    rating_elem = element.find_element(By.CSS_SELECTOR, ".a-icon-alt")
                    rating_text = rating_elem.get_attribute("textContent") if rating_elem else "0"
                    rating = float(re.search(r'(\d+\.?\d*)', rating_text).group(1)) if re.search(r'(\d+\.?\d*)', rating_text) else 0.0
                    
                    reviews_elem = element.find_element(By.CSS_SELECTOR, ".a-size-base")
                    reviews_text = reviews_elem.text if reviews_elem else "0"
                    reviews_count = int(re.sub(r'[^\d]', '', reviews_text)) if reviews_text else 0
                    
                    product = CompetitorProduct(
                        name=name,
                        brand=self._extract_brand_from_name(name),
                        price=price,
                        url=url,
                        description=name,  # Limited description from search results
                        reviews_count=reviews_count,
                        average_rating=rating,
                        availability="In Stock",  # Assumption for search results
                        image_urls=[],
                        features=[],
                        scraped_date=datetime.now()
                    )
                    
                    products.append(product)
                    
                except Exception as e:
                    logger.debug(f"Error extracting Amazon product: {e}")
                    continue
            
            driver.quit()
            
        except Exception as e:
            logger.error(f"Error scraping Amazon: {e}")
        
        return products
    
    async def _scrape_nordstrom(self, query: str, max_results: int) -> List[CompetitorProduct]:
        """Scrape Nordstrom for competitor products"""
        products = []
        
        try:
            search_url = f"https://www.nordstrom.com/sr?keyword={query.replace(' ', '%20')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find product containers
                    product_containers = soup.find_all('div', class_='_3oBa9')[:max_results]
                    
                    for container in product_containers:
                        try:
                            # Extract product information
                            name_elem = container.find('h3')
                            name = name_elem.get_text(strip=True) if name_elem else "Unknown"
                            
                            price_elem = container.find('span', class_='_2gWd5')
                            price_text = price_elem.get_text(strip=True) if price_elem else "$0"
                            price = float(re.sub(r'[^\d.]', '', price_text)) if price_text else 0.0
                            
                            link_elem = container.find('a')
                            url = "https://nordstrom.com" + link_elem.get('href') if link_elem else ""
                            
                            brand = self._extract_brand_from_name(name)
                            
                            product = CompetitorProduct(
                                name=name,
                                brand=brand,
                                price=price,
                                url=url,
                                description=name,
                                reviews_count=0,  # Not available in search results
                                average_rating=0.0,
                                availability="Available",
                                image_urls=[],
                                features=[],
                                scraped_date=datetime.now()
                            )
                            
                            products.append(product)
                            
                        except Exception as e:
                            logger.debug(f"Error extracting Nordstrom product: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Error scraping Nordstrom: {e}")
        
        return products
    
    async def _scrape_saks(self, query: str, max_results: int) -> List[CompetitorProduct]:
        """Scrape Saks Fifth Avenue for competitor products"""
        # Implementation similar to other scrapers
        # This is a placeholder - would implement actual Saks scraping logic
        return []
    
    async def _scrape_net_a_porter(self, query: str, max_results: int) -> List[CompetitorProduct]:
        """Scrape Net-a-Porter for competitor products"""
        # Implementation similar to other scrapers
        # This is a placeholder - would implement actual Net-a-Porter scraping logic
        return []
    
    async def _scrape_farfetch(self, query: str, max_results: int) -> List[CompetitorProduct]:
        """Scrape Farfetch for competitor products"""
        # Implementation similar to other scrapers
        # This is a placeholder - would implement actual Farfetch scraping logic
        return []
    
    def _extract_brand_from_name(self, product_name: str) -> str:
        """Extract brand name from product name"""
        # List of known luxury brands
        luxury_brands = [
            "Gucci", "Louis Vuitton", "Chanel", "Hermès", "Prada", "Dior",
            "Burberry", "Versace", "Saint Laurent", "Balenciaga", "Bottega Veneta",
            "Fendi", "Givenchy", "Valentino", "Tom Ford", "Alexander McQueen",
            "Stella McCartney", "Marc Jacobs", "Michael Kors", "Coach", "Kate Spade"
        ]
        
        name_lower = product_name.lower()
        for brand in luxury_brands:
            if brand.lower() in name_lower:
                return brand
        
        # If no known brand found, extract first word as potential brand
        words = product_name.split()
        return words[0] if words else "Unknown"
    
    def _analyze_pricing(self, products: List[CompetitorProduct]) -> Dict[str, float]:
        """Analyze pricing patterns in competitor products"""
        if not products:
            return {}
        
        prices = [p.price for p in products if p.price > 0]
        
        if not prices:
            return {}
        
        return {
            "min_price": min(prices),
            "max_price": max(prices),
            "avg_price": sum(prices) / len(prices),
            "median_price": sorted(prices)[len(prices)//2],
            "price_std": pd.Series(prices).std(),
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
        trends = []
        
        # Analyze product names and descriptions for trend keywords
        all_text = " ".join([f"{p.name} {p.description}" for p in products]).lower()
        
        trend_keywords = {
            "sustainable": "Sustainability focus",
            "eco-friendly": "Eco-consciousness",
            "recycled": "Circular fashion",
            "organic": "Organic materials",
            "minimalist": "Minimalist design",
            "oversized": "Oversized silhouettes",
            "vintage": "Vintage inspiration",
            "tech": "Technology integration",
            "smart": "Smart features",
            "customizable": "Personalization"
        }
        
        for keyword, trend in trend_keywords.items():
            if keyword in all_text:
                trends.append(trend)
        
        return trends[:5]  # Return top 5 trends
    
    def _identify_top_brands(self, products: List[CompetitorProduct]) -> List[str]:
        """Identify top brands by product count"""
        brand_counts = {}
        
        for product in products:
            brand = product.brand
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        # Sort by count and return top brands
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [brand for brand, count in sorted_brands[:10]]
    
    def _calculate_average_rating(self, products: List[CompetitorProduct]) -> float:
        """Calculate average rating across all products"""
        ratings = [p.average_rating for p in products if p.average_rating > 0]
        
        if not ratings:
            return 0.0
        
        return sum(ratings) / len(ratings)
    
    def get_market_summary(self, result: MarketResearchResult) -> Dict:
        """Get a summary of market research results"""
        return {
            "category": result.category,
            "total_products": result.total_products_found,
            "price_range": f"${result.price_analysis.get('min_price', 0):.0f} - ${result.price_analysis.get('max_price', 0):.0f}",
            "avg_price": f"${result.price_analysis.get('avg_price', 0):.0f}",
            "top_brands": result.top_brands[:5],
            "key_trends": result.market_trends[:3],
            "overall_sentiment": result.sentiment_analysis.get('overall_sentiment', 0),
            "data_sources": result.data_sources
        }
    
    def find_similar_products(self, product_description: str, max_results: int = 20) -> List[CompetitorProduct]:
        """
        Find similar products based on user description
        
        Args:
            product_description: User's product description
            max_results: Maximum number of products to find
        
        Returns:
            List of similar competitor products
        """
        logger.info(f"Finding similar products for: {product_description}")
        
        # Extract key terms from product description
        search_terms = self._extract_search_terms(product_description)
        
        all_products = []
        
        # Search across multiple platforms
        platforms = ['amazon', 'zalando', 'farfetch', 'net-a-porter']
        
        for platform in platforms:
            try:
                products = self._search_platform(platform, search_terms, max_results // len(platforms))
                all_products.extend(products)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error searching {platform}: {e}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_products = self._deduplicate_products(all_products)
        
        return unique_products[:max_results]
    
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
    
    def _search_platform(self, platform: str, search_terms: List[str], max_results: int) -> List[CompetitorProduct]:
        """Search a specific platform for products"""
        
        if platform == 'amazon':
            return self._search_amazon(search_terms, max_results)
        elif platform == 'zalando':
            return self._search_zalando(search_terms, max_results)
        elif platform == 'farfetch':
            return self._search_farfetch(search_terms, max_results)
        elif platform == 'net-a-porter':
            return self._search_net_a_porter(search_terms, max_results)
        else:
            return []
    
    def _search_amazon(self, search_terms: List[str], max_results: int) -> List[CompetitorProduct]:
        """Search Amazon for fashion products"""
        products = []
        
        try:
            # Create search query
            query = ' '.join(search_terms[:3])  # Use top 3 terms
            
            # Use requests with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Amazon search URL
            url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}&ref=sr_pg_1"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find product containers
                product_containers = soup.find_all('div', {'data-component-type': 's-search-result'})
                
                for container in product_containers[:max_results]:
                    try:
                        # Extract product info
                        name_elem = container.find('h2', class_='a-size-mini')
                        if name_elem:
                            name = name_elem.get_text(strip=True)
                        else:
                            continue
                        
                        price_elem = container.find('span', class_='a-price-whole')
                        price = 0.0
                        if price_elem:
                            price_text = price_elem.get_text(strip=True).replace(',', '')
                            try:
                                price = float(price_text)
                            except:
                                price = 0.0
                        
                        # Get product URL
                        link_elem = container.find('a', class_='a-link-normal')
                        product_url = ""
                        if link_elem:
                            product_url = "https://www.amazon.com" + link_elem.get('href', '')
                        
                        # Get rating info
                        rating_elem = container.find('span', class_='a-icon-alt')
                        rating = 0.0
                        reviews_count = 0
                        if rating_elem:
                            rating_text = rating_elem.get_text()
                            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                            if rating_match:
                                rating = float(rating_match.group(1))
                        
                        # Create product object
                        product = CompetitorProduct(
                            name=name,
                            brand="Amazon Product",  # Would need additional parsing
                            price=price,
                            url=product_url,
                            description="",
                            reviews_count=reviews_count,
                            average_rating=rating,
                            availability="In Stock",
                            image_urls=[],
                            features=[],
                            scraped_date=datetime.now()
                        )
                        
                        products.append(product)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing Amazon product: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error searching Amazon: {e}")
        
        return products
    
    def analyze_competitor_sentiment(self, products: List[CompetitorProduct]) -> Dict[str, float]:
        """
        Analyze sentiment across competitor products
        
        Args:
            products: List of competitor products to analyze
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if not products:
            return {"overall_sentiment": 0.0, "total_products": 0}
        
        total_sentiment = 0.0
        sentiment_count = 0
        high_rated_count = 0
        low_rated_count = 0
        
        for product in products:
            if product.average_rating > 0:
                # Convert rating to sentiment score (1-5 scale to -1 to 1 scale)
                sentiment_score = (product.average_rating - 3) / 2
                total_sentiment += sentiment_score
                sentiment_count += 1
                
                if product.average_rating >= 4.0:
                    high_rated_count += 1
                elif product.average_rating <= 2.0:
                    low_rated_count += 1
        
        avg_sentiment = total_sentiment / sentiment_count if sentiment_count > 0 else 0.0
        
        return {
            "overall_sentiment": avg_sentiment,
            "total_products": len(products),
            "high_rated_percentage": high_rated_count / len(products) if products else 0,
            "low_rated_percentage": low_rated_count / len(products) if products else 0,
            "sentiment_count": sentiment_count
        }
