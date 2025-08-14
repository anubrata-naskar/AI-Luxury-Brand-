"""
Database models and utilities for storing analysis results
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from typing import Dict, List, Optional

Base = declarative_base()

class ProductAnalysisRecord(Base):
    """Database model for product analysis results"""
    __tablename__ = "product_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), index=True)
    brand = Column(String(100), index=True)
    category = Column(String(50), index=True)
    price = Column(Float)
    
    # Analysis results
    overall_score = Column(Float)
    trend_alignment_score = Column(Float)
    value_score = Column(Float)
    confidence_level = Column(Float)
    
    # JSON fields for complex data
    market_positioning = Column(Text)
    competitive_landscape = Column(JSON)
    unique_selling_points = Column(JSON)
    key_features = Column(JSON)
    missing_features = Column(JSON)
    target_demographics = Column(JSON)
    seasonal_demand = Column(JSON)
    marketing_recommendations = Column(JSON)
    
    # Metadata
    analysis_date = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))
    
class MarketResearchRecord(Base):
    """Database model for market research results"""
    __tablename__ = "market_research"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), index=True)
    search_query = Column(String(255))
    total_products_found = Column(Integer)
    
    # Analysis results
    price_analysis = Column(JSON)
    sentiment_analysis = Column(JSON)
    market_trends = Column(JSON)
    top_brands = Column(JSON)
    average_rating = Column(Float)
    
    # Metadata
    research_date = Column(DateTime, default=datetime.utcnow)
    data_sources = Column(JSON)

class DatabaseManager:
    """Database manager for the fashion analysis system"""
    
    def __init__(self, database_url: str = "sqlite:///./data/fashion_analysis.db"):
        """Initialize database connection"""
        self.engine = create_engine(database_url)
        Base.metadata.create_all(bind=self.engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session = SessionLocal()
    
    def save_product_analysis(self, analysis_data: Dict) -> int:
        """Save product analysis to database"""
        record = ProductAnalysisRecord(
            product_name=analysis_data.get('product_name'),
            brand=analysis_data.get('brand'),
            category=analysis_data.get('category'),
            price=analysis_data.get('price'),
            overall_score=analysis_data.get('overall_score'),
            trend_alignment_score=analysis_data.get('trend_alignment_score'),
            value_score=analysis_data.get('value_score'),
            confidence_level=analysis_data.get('confidence_level'),
            market_positioning=analysis_data.get('market_positioning'),
            competitive_landscape=analysis_data.get('competitive_landscape'),
            unique_selling_points=analysis_data.get('unique_selling_points'),
            key_features=analysis_data.get('key_features'),
            missing_features=analysis_data.get('missing_features'),
            target_demographics=analysis_data.get('target_demographics'),
            seasonal_demand=analysis_data.get('seasonal_demand'),
            marketing_recommendations=analysis_data.get('marketing_recommendations'),
            model_version=analysis_data.get('model_version')
        )
        
        self.session.add(record)
        self.session.commit()
        
        return record.id
    
    def save_market_research(self, research_data: Dict) -> int:
        """Save market research to database"""
        record = MarketResearchRecord(
            category=research_data.get('category'),
            search_query=research_data.get('search_query'),
            total_products_found=research_data.get('total_products_found'),
            price_analysis=research_data.get('price_analysis'),
            sentiment_analysis=research_data.get('sentiment_analysis'),
            market_trends=research_data.get('market_trends'),
            top_brands=research_data.get('top_brands'),
            average_rating=research_data.get('average_rating'),
            data_sources=research_data.get('data_sources')
        )
        
        self.session.add(record)
        self.session.commit()
        
        return record.id
    
    def get_product_analyses(self, limit: int = 100) -> List[Dict]:
        """Get recent product analyses"""
        records = self.session.query(ProductAnalysisRecord)\
                               .order_by(ProductAnalysisRecord.analysis_date.desc())\
                               .limit(limit).all()
        
        return [self._product_record_to_dict(record) for record in records]
    
    def get_market_research(self, limit: int = 100) -> List[Dict]:
        """Get recent market research"""
        records = self.session.query(MarketResearchRecord)\
                               .order_by(MarketResearchRecord.research_date.desc())\
                               .limit(limit).all()
        
        return [self._research_record_to_dict(record) for record in records]
    
    def _product_record_to_dict(self, record: ProductAnalysisRecord) -> Dict:
        """Convert product analysis record to dictionary"""
        return {
            'id': record.id,
            'product_name': record.product_name,
            'brand': record.brand,
            'category': record.category,
            'price': record.price,
            'overall_score': record.overall_score,
            'trend_alignment_score': record.trend_alignment_score,
            'value_score': record.value_score,
            'confidence_level': record.confidence_level,
            'analysis_date': record.analysis_date.isoformat() if record.analysis_date else None,
            'model_version': record.model_version
        }
    
    def _research_record_to_dict(self, record: MarketResearchRecord) -> Dict:
        """Convert market research record to dictionary"""
        return {
            'id': record.id,
            'category': record.category,
            'search_query': record.search_query,
            'total_products_found': record.total_products_found,
            'average_rating': record.average_rating,
            'research_date': record.research_date.isoformat() if record.research_date else None,
            'top_brands': record.top_brands,
            'market_trends': record.market_trends
        }
