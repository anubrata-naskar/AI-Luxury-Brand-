"""
Test suite for the AI Fashion Analysis System
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.product_analyzer import ProductAnalyzer, ProductInsight
from src.agents.market_agent import MarketResearchAgent
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.text_processing import TextProcessor

class TestProductAnalyzer:
    """Test cases for ProductAnalyzer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = ProductAnalyzer()
        self.sample_product = {
            "name": "Test Cashmere Sweater",
            "brand": "Test Brand",
            "price": 299.99,
            "description": "Premium cashmere sweater with elegant design",
            "category": "sweaters"
        }
    
    def test_product_categorization(self):
        """Test product categorization"""
        result = self.analyzer._categorize_product("Cashmere Sweater", "soft wool sweater")
        assert result == "sweaters"
        
        result = self.analyzer._categorize_product("Leather Handbag", "designer bag")
        assert result == "bags"
    
    def test_trend_alignment_calculation(self):
        """Test trend alignment scoring"""
        score = self.analyzer._calculate_trend_alignment(
            "sustainable organic cotton dress", 
            "dresses", 
            ""
        )
        assert 0 <= score <= 10
        assert score > 5  # Should score higher due to sustainability keywords
    
    def test_value_score_calculation(self):
        """Test value score calculation"""
        score = self.analyzer._calculate_value_score(
            200, "premium cotton high-quality", "TestBrand", "sweaters"
        )
        assert 0 <= score <= 10
    
    def test_feature_extraction(self):
        """Test feature extraction from description"""
        features = self.analyzer._extract_key_features(
            "100% cashmere hand-knitted machine-washable"
        )
        assert "hand-knitted" in features or "machine-washable" in features

class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = SentimentAnalyzer(use_transformer=False)
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        result = self.analyzer.analyze_text("This product is absolutely beautiful and amazing!")
        assert result['compound'] > 0
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        result = self.analyzer.analyze_text("This product is terrible and disappointing!")
        assert result['compound'] < 0
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        result = self.analyzer.analyze_text("This product is okay.")
        assert abs(result['compound']) < 0.5
    
    def test_fashion_specific_sentiment(self):
        """Test fashion-specific sentiment analysis"""
        result = self.analyzer._analyze_fashion_sentiment("gorgeous elegant stylish")
        assert result['positive'] > 0
    
    def test_emotion_detection(self):
        """Test emotion detection"""
        emotions = self.analyzer.detect_emotion("I love this beautiful dress!")
        assert 'joy' in emotions
        assert emotions['joy'] > 0

class TestTextProcessor:
    """Test cases for TextProcessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = TextProcessor()
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        cleaned = self.processor.clean_text("  HELLO WORLD!!!  ")
        assert cleaned == "hello world"
    
    def test_keyword_extraction(self):
        """Test keyword extraction"""
        keywords = self.processor.extract_keywords(
            "This beautiful cashmere sweater is made with premium materials"
        )
        assert len(keywords) > 0
        assert any(word in ['cashmere', 'beautiful', 'premium'] for word in keywords)
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        entities = self.processor.extract_entities(
            "This red cashmere sweater is size large"
        )
        assert 'colors' in entities
        assert 'materials' in entities
        assert 'sizes' in entities
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        features = self.processor.extract_features(
            "Made of premium wool with button closure"
        )
        assert len(features) > 0

class TestMarketResearchAgent:
    """Test cases for MarketResearchAgent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.agent = MarketResearchAgent()
    
    def test_search_query_generation(self):
        """Test search query generation"""
        queries = self.agent._generate_search_queries("luxury cashmere sweater", "sweaters")
        assert len(queries) > 0
        assert any("luxury" in query for query in queries)
    
    def test_brand_extraction(self):
        """Test brand extraction from product names"""
        brand = self.agent._extract_brand_from_name("Gucci Leather Handbag")
        assert brand == "Gucci"
    
    def test_pricing_analysis(self):
        """Test pricing analysis"""
        from src.agents.market_agent import CompetitorProduct
        from datetime import datetime
        
        products = [
            CompetitorProduct(
                name="Test Product 1",
                brand="Brand A",
                price=100.0,
                url="",
                description="",
                reviews_count=10,
                average_rating=4.5,
                availability="In Stock",
                image_urls=[],
                features=[],
                scraped_date=datetime.now()
            ),
            CompetitorProduct(
                name="Test Product 2",
                brand="Brand B",
                price=200.0,
                url="",
                description="",
                reviews_count=20,
                average_rating=4.0,
                availability="In Stock",
                image_urls=[],
                features=[],
                scraped_date=datetime.now()
            )
        ]
        
        analysis = self.agent._analyze_pricing(products)
        assert 'min_price' in analysis
        assert 'max_price' in analysis
        assert 'avg_price' in analysis
        assert analysis['min_price'] == 100.0
        assert analysis['max_price'] == 200.0

# Integration tests
class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis workflow"""
        analyzer = ProductAnalyzer()
        
        sample_product = {
            "name": "Luxury Cashmere Sweater",
            "brand": "Test Brand",
            "price": 450.0,
            "description": "Premium 100% cashmere sweater with elegant design and superior comfort",
            "category": "sweaters"
        }
        
        # This should not raise any exceptions
        result = analyzer.analyze_product(sample_product)
        
        # Verify result structure
        assert hasattr(result, 'product_name')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'trend_alignment_score')
        assert hasattr(result, 'value_score')
        assert 0 <= result.overall_score <= 10
        assert 0 <= result.trend_alignment_score <= 10
        assert 0 <= result.value_score <= 10

if __name__ == "__main__":
    pytest.main([__file__])
