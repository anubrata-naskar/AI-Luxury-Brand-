"""
FastAPI application for the luxury fashion analysis system
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import json

from ..analysis.product_analyzer import ProductAnalyzer, ProductInsight
from ..agents.market_agent import MarketResearchAgent, MarketResearchResult
from ..models.gemini_fashion import FashionGemini

# Pydantic models for API requests/responses
class ProductAnalysisRequest(BaseModel):
    name: str
    brand: str
    price: float
    description: str
    category: Optional[str] = None

class MarketResearchRequest(BaseModel):
    product_description: str
    category: str
    max_products: Optional[int] = 50
    min_price: Optional[float] = None
    max_price: Optional[float] = None

class AnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: datetime
    processing_time: Optional[float] = None

# Initialize FastAPI app
app = FastAPI(
    title="AI Luxury Fashion Analysis API",
    description="Comprehensive AI-powered fashion product analysis and market research",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialize on startup)
product_analyzer: Optional[ProductAnalyzer] = None
market_agent: Optional[MarketResearchAgent] = None
fashion_model: Optional[FashionGemini] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and analyzers on startup"""
    global product_analyzer, market_agent, fashion_model
    
    try:
        # Initialize components
        product_analyzer = ProductAnalyzer()  # Will use GEMINI_API_KEY from environment
        market_agent = MarketResearchAgent()
        
        # Try to initialize the fashion model (might fail if API key not available)
        try:
            fashion_model = FashionGemini()
        except Exception as e:
            print(f"Warning: Could not initialize fashion model: {e}")
            fashion_model = None
        
        print("API initialized successfully")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Luxury Fashion Analysis API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "product_analyzer": product_analyzer is not None,
            "market_agent": market_agent is not None,
            "fashion_model": fashion_model is not None
        },
        "timestamp": datetime.now()
    }

@app.post("/analyze/product", response_model=AnalysisResponse)
async def analyze_product(request: ProductAnalysisRequest):
    """
    Analyze a fashion product and provide comprehensive insights
    """
    start_time = datetime.now()
    
    try:
        if not product_analyzer:
            raise HTTPException(status_code=500, detail="Product analyzer not initialized")
        
        # Convert request to dict
        product_data = {
            "name": request.name,
            "brand": request.brand,
            "price": request.price,
            "description": request.description,
            "category": request.category
        }
        
        # Perform analysis
        insight = product_analyzer.analyze_product(product_data)
        
        # Convert to serializable format
        insight_dict = {
            "product_name": insight.product_name,
            "brand": insight.brand,
            "category": insight.category,
            "price": insight.price,
            "market_positioning": insight.market_positioning,
            "competitive_landscape": insight.competitive_landscape,
            "unique_selling_points": insight.unique_selling_points,
            "trend_alignment_score": insight.trend_alignment_score,
            "trend_category": insight.trend_category.value,
            "style_longevity": insight.style_longevity,
            "price_category": insight.price_category.value,
            "value_score": insight.value_score,
            "price_recommendations": insight.price_recommendations,
            "key_features": insight.key_features,
            "missing_features": insight.missing_features,
            "feature_gaps": insight.feature_gaps,
            "target_demographics": insight.target_demographics,
            "customer_personas": insight.customer_personas,
            "seasonal_demand": insight.seasonal_demand,
            "peak_seasons": insight.peak_seasons,
            "overall_score": insight.overall_score,
            "confidence_level": insight.confidence_level,
            "marketing_recommendations": insight.marketing_recommendations,
            "product_improvements": insight.product_improvements,
            "analysis_date": insight.analysis_date.isoformat(),
            "model_version": insight.model_version
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            data=insight_dict,
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/research/market", response_model=AnalysisResponse)
async def research_market(request: MarketResearchRequest, background_tasks: BackgroundTasks):
    """
    Perform market research and competitor analysis
    """
    start_time = datetime.now()
    
    try:
        if not market_agent:
            raise HTTPException(status_code=500, detail="Market research agent not initialized")
        
        # Set price range if provided
        price_range = None
        if request.min_price is not None and request.max_price is not None:
            price_range = (request.min_price, request.max_price)
        
        # Perform market research
        result = await market_agent.research_competitors(
            product_description=request.product_description,
            category=request.category,
            max_products=request.max_products,
            price_range=price_range
        )
        
        # Convert to serializable format
        result_dict = {
            "category": result.category,
            "search_query": result.search_query,
            "total_products_found": result.total_products_found,
            "competitor_products": [
                {
                    "name": p.name,
                    "brand": p.brand,
                    "price": p.price,
                    "url": p.url,
                    "description": p.description,
                    "reviews_count": p.reviews_count,
                    "average_rating": p.average_rating,
                    "availability": p.availability,
                    "features": p.features,
                    "scraped_date": p.scraped_date.isoformat()
                }
                for p in result.competitor_products[:20]  # Limit for response size
            ],
            "price_analysis": result.price_analysis,
            "sentiment_analysis": result.sentiment_analysis,
            "market_trends": result.market_trends,
            "top_brands": result.top_brands,
            "average_rating": result.average_rating,
            "research_date": result.research_date.isoformat(),
            "data_sources": result.data_sources
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            data=result_dict,
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market research failed: {str(e)}")

@app.post("/analyze/trends")
async def analyze_trends(category: str, timeframe: str = "next_season"):
    """
    Analyze fashion trends for a specific category and timeframe
    """
    try:
        if not fashion_model:
            raise HTTPException(status_code=503, detail="Fashion model not available")
        
        # Generate trend predictions
        predictions = fashion_model.predict_trends(timeframe)
        
        return {
            "status": "success",
            "data": {
                "category": category,
                "timeframe": timeframe,
                "trend_predictions": predictions,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@app.post("/analyze/market-insights")
async def get_market_insights(category: str, brand: Optional[str] = None):
    """
    Get market insights for a specific category or brand
    """
    try:
        if not fashion_model:
            raise HTTPException(status_code=503, detail="Fashion model not available")
        
        # Generate market insights
        insights = fashion_model.get_market_insights(category, brand)
        
        return {
            "status": "success",
            "data": {
                "category": category,
                "brand": brand,
                "market_insights": insights,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market insights failed: {str(e)}")

@app.get("/categories")
async def get_supported_categories():
    """
    Get list of supported product categories
    """
    categories = [
        "dresses", "sweaters", "jeans", "shoes", "bags", "accessories",
        "coats", "jackets", "blouses", "skirts", "pants", "jewelry"
    ]
    
    return {
        "status": "success",
        "data": {
            "categories": categories,
            "total_count": len(categories)
        }
    }

@app.get("/brands")
async def get_luxury_brands():
    """
    Get list of luxury brands in the system
    """
    brands = [
        "Gucci", "Louis Vuitton", "Chanel", "Hermès", "Prada", "Dior",
        "Burberry", "Versace", "Saint Laurent", "Balenciaga", "Bottega Veneta",
        "Fendi", "Givenchy", "Valentino", "Tom Ford", "Alexander McQueen"
    ]
    
    return {
        "status": "success",
        "data": {
            "luxury_brands": brands,
            "total_count": len(brands)
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "status": "error",
        "message": "Endpoint not found",
        "timestamp": datetime.now()
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
