"""
Product Analysis Engine
Provides comprehensive analysis of fashion products using AI models
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from ..models.gemini_fashion import FashionGemini
from ..utils.text_processing import TextProcessor
from ..utils.sentiment_analyzer import SentimentAnalyzer

class TrendCategory(Enum):
    EMERGING = "emerging"
    MAINSTREAM = "mainstream"
    DECLINING = "declining"
    CLASSIC = "classic"

class PriceCategory(Enum):
    BUDGET = "budget"
    MID_RANGE = "mid_range"
    PREMIUM = "premium"
    LUXURY = "luxury"
    ULTRA_LUXURY = "ultra_luxury"

@dataclass
class ProductInsight:
    """Structure for product analysis results"""
    product_name: str
    brand: str
    category: str
    price: float
    
    # Market positioning
    market_positioning: str
    competitive_landscape: List[str]
    unique_selling_points: List[str]
    
    # Trend analysis
    trend_alignment_score: float  # 0-10
    trend_category: TrendCategory
    style_longevity: str
    
    # Price analysis
    price_category: PriceCategory
    value_score: float  # 0-10
    price_recommendations: List[str]
    
    # Feature analysis
    key_features: List[str]
    missing_features: List[str]
    feature_gaps: List[str]
    
    # Target audience
    target_demographics: Dict[str, str]
    customer_personas: List[str]
    
    # Seasonal analysis
    seasonal_demand: Dict[str, float]
    peak_seasons: List[str]
    
    # Overall scores
    overall_score: float
    confidence_level: float
    
    # Recommendations
    marketing_recommendations: List[str]
    product_improvements: List[str]
    
    # Metadata
    analysis_date: datetime
    model_version: str

class ProductAnalyzer:
    """Main product analysis engine"""
    
    def __init__(self, api_key: str = None):
        """Initialize the product analyzer"""
        self.model = FashionGemini(api_key) if api_key or os.getenv('GEMINI_API_KEY') else None
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Load category mappings and price ranges
        self.category_mappings = self._load_category_mappings()
        self.price_ranges = self._load_price_ranges()
        
    def analyze_product(self, product_data: Dict) -> ProductInsight:
        """
        Comprehensive product analysis
        
        Args:
            product_data: Dictionary containing product information
        
        Returns:
            ProductInsight object with complete analysis
        """
        logger.info(f"Starting analysis for product: {product_data.get('name', 'Unknown')}")
        
        try:
            # Extract basic information
            product_name = product_data.get('name', '')
            brand = product_data.get('brand', '')
            price = float(product_data.get('price', 0))
            description = product_data.get('description', '')
            category = self._categorize_product(product_name, description)
            
            # Generate AI analysis if model is available
            ai_analysis = ""
            if self.model:
                full_description = f"Product: {product_name}, Brand: {brand}, Price: ${price}, Description: {description}"
                ai_analysis = self.model.analyze_product(full_description, "comprehensive")
            
            # Market positioning analysis
            market_positioning = self._analyze_market_positioning(brand, price, category, ai_analysis)
            competitive_landscape = self._identify_competitors(brand, category)
            unique_selling_points = self._extract_usp(description, ai_analysis)
            
            # Trend analysis
            trend_alignment_score = self._calculate_trend_alignment(description, category, ai_analysis)
            trend_category = self._categorize_trend(trend_alignment_score)
            style_longevity = self._predict_style_longevity(description, category)
            
            # Price analysis
            price_category = self._categorize_price(price, category)
            value_score = self._calculate_value_score(price, description, brand, category)
            price_recommendations = self._generate_price_recommendations(price, category, value_score)
            
            # Feature analysis
            key_features = self._extract_key_features(description)
            missing_features, feature_gaps = self._analyze_feature_gaps(category, key_features)
            
            # Target audience analysis
            target_demographics = self._analyze_target_demographics(price, brand, category, description)
            customer_personas = self._generate_customer_personas(target_demographics, price_category)
            
            # Seasonal analysis
            seasonal_demand = self._analyze_seasonal_demand(category, description)
            peak_seasons = self._identify_peak_seasons(seasonal_demand)
            
            # Calculate overall scores
            overall_score = self._calculate_overall_score(
                trend_alignment_score, value_score, len(key_features), price_category
            )
            confidence_level = self._calculate_confidence_level(ai_analysis, description)
            
            # Generate recommendations
            marketing_recommendations = self._generate_marketing_recommendations(
                target_demographics, trend_alignment_score, unique_selling_points
            )
            product_improvements = self._generate_product_improvements(
                missing_features, trend_alignment_score, value_score
            )
            
            # Create insight object
            insight = ProductInsight(
                product_name=product_name,
                brand=brand,
                category=category,
                price=price,
                market_positioning=market_positioning,
                competitive_landscape=competitive_landscape,
                unique_selling_points=unique_selling_points,
                trend_alignment_score=trend_alignment_score,
                trend_category=trend_category,
                style_longevity=style_longevity,
                price_category=price_category,
                value_score=value_score,
                price_recommendations=price_recommendations,
                key_features=key_features,
                missing_features=missing_features,
                feature_gaps=feature_gaps,
                target_demographics=target_demographics,
                customer_personas=customer_personas,
                seasonal_demand=seasonal_demand,
                peak_seasons=peak_seasons,
                overall_score=overall_score,
                confidence_level=confidence_level,
                marketing_recommendations=marketing_recommendations,
                product_improvements=product_improvements,
                analysis_date=datetime.now(),
                model_version="fashion-llama-3b-v1"
            )
            
            logger.info(f"Analysis completed for {product_name} with overall score: {overall_score:.2f}")
            return insight
            
        except Exception as e:
            logger.error(f"Error analyzing product: {e}")
            raise
    
    def _categorize_product(self, name: str, description: str) -> str:
        """Categorize product based on name and description"""
        text = f"{name} {description}".lower()
        
        category_keywords = {
            "dresses": ["dress", "gown", "frock"],
            "sweaters": ["sweater", "jumper", "pullover", "cardigan", "knitwear"],
            "jeans": ["jeans", "denim"],
            "shoes": ["shoes", "boots", "sneakers", "heels", "flats", "sandals"],
            "bags": ["bag", "handbag", "purse", "clutch", "tote", "backpack"],
            "accessories": ["belt", "scarf", "hat", "gloves", "sunglasses"],
            "coats": ["coat", "jacket", "blazer", "outerwear"],
            "blouses": ["blouse", "shirt", "top"],
            "skirts": ["skirt", "mini", "maxi"],
            "pants": ["pants", "trousers", "leggings"],
            "jewelry": ["jewelry", "necklace", "bracelet", "earrings", "ring", "watch"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return "other"
    
    def _analyze_market_positioning(self, brand: str, price: float, category: str, ai_analysis: str) -> str:
        """Analyze market positioning"""
        price_cat = self._categorize_price(price, category)
        
        positioning_map = {
            PriceCategory.ULTRA_LUXURY: "Ultra-luxury segment targeting HNW individuals",
            PriceCategory.LUXURY: "Luxury segment with premium positioning",
            PriceCategory.PREMIUM: "Premium segment with aspirational appeal",
            PriceCategory.MID_RANGE: "Mid-market with value proposition",
            PriceCategory.BUDGET: "Accessible luxury with competitive pricing"
        }
        
        base_positioning = positioning_map.get(price_cat, "Market positioning unclear")
        
        # Enhance with AI insights if available
        if ai_analysis and "positioning" in ai_analysis.lower():
            ai_positioning = self._extract_positioning_from_ai(ai_analysis)
            if ai_positioning:
                base_positioning = f"{base_positioning}. {ai_positioning}"
        
        return base_positioning
    
    def _calculate_trend_alignment(self, description: str, category: str, ai_analysis: str) -> float:
        """Calculate trend alignment score (0-10)"""
        # Current trend keywords (would be updated regularly)
        trend_keywords = {
            "sustainability": 2.0,
            "eco-friendly": 2.0,
            "organic": 1.5,
            "recycled": 1.5,
            "vintage": 1.0,
            "minimalist": 1.5,
            "oversized": 1.0,
            "cropped": 1.0,
            "wide-leg": 1.0,
            "statement": 1.0
        }
        
        text = description.lower()
        score = 5.0  # Base score
        
        # Add points for trend keywords
        for keyword, weight in trend_keywords.items():
            if keyword in text:
                score += weight
        
        # Extract AI trend score if available
        if ai_analysis:
            ai_score = self._extract_ai_trend_score(ai_analysis)
            if ai_score:
                score = (score + ai_score) / 2
        
        return min(score, 10.0)
    
    def _calculate_value_score(self, price: float, description: str, brand: str, category: str) -> float:
        """Calculate value score based on price and features"""
        # Get expected price range for category
        price_range = self.price_ranges.get(category, {"min": 50, "max": 500})
        
        # Normalize price within category
        price_ratio = price / price_range["max"]
        
        # Count valuable features
        feature_score = len(self._extract_key_features(description)) * 0.5
        
        # Brand premium factor
        luxury_brands = ["Gucci", "Louis Vuitton", "Chanel", "Hermès", "Prada"]
        brand_factor = 2.0 if brand in luxury_brands else 1.0
        
        # Calculate value score
        base_score = 5.0
        if price_ratio < 0.5:  # Good value
            base_score += 2.0
        elif price_ratio > 2.0:  # Expensive
            base_score -= 1.0
        
        value_score = base_score + feature_score * brand_factor
        
        return min(max(value_score, 0.0), 10.0)
    
    def _extract_key_features(self, description: str) -> List[str]:
        """Extract key product features from description"""
        features = []
        
        # Material features
        materials = re.findall(r'\b(?:100%\s+)?(\w+(?:\s+\w+)*)\b(?=\s+(?:fabric|material|leather|cotton|silk|wool|cashmere|linen))', description.lower())
        features.extend([f"{mat} material" for mat in materials])
        
        # Construction features
        construction_keywords = ["hand-made", "handcrafted", "machine-washable", "wrinkle-free", "stretch"]
        features.extend([kw for kw in construction_keywords if kw in description.lower()])
        
        # Style features
        style_keywords = ["classic", "modern", "vintage", "contemporary", "timeless"]
        features.extend([kw for kw in style_keywords if kw in description.lower()])
        
        return list(set(features))  # Remove duplicates
    
    def _analyze_feature_gaps(self, category: str, current_features: List[str]) -> Tuple[List[str], List[str]]:
        """Analyze missing features and gaps"""
        expected_features = {
            "dresses": ["lining", "adjustable straps", "pockets", "zipper closure"],
            "sweaters": ["machine washable", "pill-resistant", "breathable"],
            "shoes": ["cushioned insole", "slip-resistant sole", "arch support"],
            "bags": ["multiple compartments", "adjustable strap", "dust bag"]
        }
        
        category_features = expected_features.get(category, [])
        current_lower = [f.lower() for f in current_features]
        
        missing_features = [f for f in category_features if f not in current_lower]
        
        # Identify potential gaps
        feature_gaps = []
        if "sustainable" not in " ".join(current_lower):
            feature_gaps.append("Sustainability features")
        if "technology" not in " ".join(current_lower):
            feature_gaps.append("Tech integration opportunities")
        
        return missing_features, feature_gaps
    
    def _analyze_seasonal_demand(self, category: str, description: str) -> Dict[str, float]:
        """Analyze seasonal demand patterns"""
        seasonal_patterns = {
            "coats": {"spring": 0.2, "summer": 0.1, "fall": 0.4, "winter": 0.3},
            "sweaters": {"spring": 0.3, "summer": 0.1, "fall": 0.3, "winter": 0.3},
            "dresses": {"spring": 0.3, "summer": 0.4, "fall": 0.2, "winter": 0.1},
            "shoes": {"spring": 0.25, "summer": 0.25, "fall": 0.25, "winter": 0.25}
        }
        
        base_pattern = seasonal_patterns.get(category, {"spring": 0.25, "summer": 0.25, "fall": 0.25, "winter": 0.25})
        
        # Adjust based on description
        desc_lower = description.lower()
        if "summer" in desc_lower or "lightweight" in desc_lower:
            base_pattern["summer"] += 0.1
            base_pattern["winter"] -= 0.1
        elif "winter" in desc_lower or "warm" in desc_lower:
            base_pattern["winter"] += 0.1
            base_pattern["summer"] -= 0.1
        
        return base_pattern
    
    def _generate_marketing_recommendations(self, demographics: Dict, trend_score: float, usps: List[str]) -> List[str]:
        """Generate marketing recommendations"""
        recommendations = []
        
        if trend_score > 7:
            recommendations.append("Emphasize trend-forward positioning in marketing")
        elif trend_score < 4:
            recommendations.append("Focus on classic/timeless appeal rather than trends")
        
        if "luxury" in str(demographics.values()).lower():
            recommendations.append("Target luxury lifestyle publications and influencers")
        
        if len(usps) > 2:
            recommendations.append("Highlight unique selling points in product storytelling")
        
        recommendations.append("Leverage social media for visual product showcasing")
        
        return recommendations
    
    def _load_category_mappings(self) -> Dict:
        """Load category mapping configurations"""
        # This would typically load from a config file
        return {
            "clothing": ["dresses", "sweaters", "jeans", "coats", "blouses", "skirts", "pants"],
            "footwear": ["shoes"],
            "accessories": ["bags", "jewelry", "accessories"]
        }
    
    def _load_price_ranges(self) -> Dict:
        """Load price range configurations for different categories"""
        return {
            "dresses": {"min": 100, "max": 2000},
            "sweaters": {"min": 80, "max": 800},
            "jeans": {"min": 50, "max": 500},
            "shoes": {"min": 100, "max": 1500},
            "bags": {"min": 200, "max": 5000},
            "jewelry": {"min": 50, "max": 10000}
        }
    
    def _categorize_price(self, price: float, category: str) -> PriceCategory:
        """Categorize price level"""
        ranges = self.price_ranges.get(category, {"min": 50, "max": 500})
        
        if price < ranges["min"] * 0.5:
            return PriceCategory.BUDGET
        elif price < ranges["min"]:
            return PriceCategory.MID_RANGE
        elif price < ranges["max"]:
            return PriceCategory.PREMIUM
        elif price < ranges["max"] * 2:
            return PriceCategory.LUXURY
        else:
            return PriceCategory.ULTRA_LUXURY
    
    def _identify_competitors(self, brand: str, category: str) -> List[str]:
        """Identify key competitors for the product"""
        
        # Define competitor mappings by category
        competitor_mappings = {
            "luxury handbags": ["Louis Vuitton", "Chanel", "Hermès", "Gucci", "Prada"],
            "luxury clothing": ["Gucci", "Prada", "Dior", "Saint Laurent", "Versace"],
            "accessories": ["Cartier", "Tiffany & Co.", "Van Cleef & Arpels", "Bulgari"],
            "footwear": ["Louboutin", "Jimmy Choo", "Manolo Blahnik", "Gucci", "Prada"],
            "jewelry": ["Tiffany & Co.", "Cartier", "Van Cleef & Arpels", "Harry Winston"],
            "watches": ["Rolex", "Patek Philippe", "Audemars Piguet", "Cartier"],
            "cosmetics": ["Chanel", "Dior", "Tom Ford", "La Mer", "La Prairie"],
            "default": ["Gucci", "Louis Vuitton", "Chanel", "Prada", "Hermès"]
        }
        
        # Get competitors for the category
        category_lower = category.lower()
        competitors = []
        
        for key, brands in competitor_mappings.items():
            if key in category_lower:
                competitors = brands
                break
        
        if not competitors:
            competitors = competitor_mappings["default"]
        
        # Remove the current brand from competitors
        competitors = [comp for comp in competitors if comp.lower() != brand.lower()]
        
        return competitors[:5]  # Return top 5 competitors
    
    def _extract_usp(self, description: str, ai_analysis: str) -> List[str]:
        """Extract unique selling points from description and AI analysis"""
        
        usps = []
        description_lower = description.lower()
        
        # Look for quality indicators
        quality_keywords = ["premium", "luxury", "handcrafted", "artisan", "exclusive", 
                          "limited edition", "bespoke", "custom", "high-quality"]
        for keyword in quality_keywords:
            if keyword in description_lower:
                usps.append(f"Premium {keyword.replace('_', ' ')}")
        
        # Look for material mentions
        material_keywords = ["leather", "silk", "cashmere", "gold", "silver", "diamond", 
                           "platinum", "titanium", "carbon fiber"]
        for material in material_keywords:
            if material in description_lower:
                usps.append(f"Premium {material}")
        
        # Look for sustainability
        sustainability_keywords = ["sustainable", "eco-friendly", "organic", "recycled", "vegan"]
        for keyword in sustainability_keywords:
            if keyword in description_lower:
                usps.append("Sustainable design")
                break
        
        # Extract from AI analysis if available
        if ai_analysis:
            ai_lower = ai_analysis.lower()
            if "unique" in ai_lower:
                usps.append("Unique design elements")
            if "innovative" in ai_lower:
                usps.append("Innovative features")
            if "heritage" in ai_lower or "tradition" in ai_lower:
                usps.append("Heritage craftsmanship")
        
        # Remove duplicates and limit to top 5
        usps = list(set(usps))
        return usps[:5]
    
    def _categorize_trend(self, trend_score: float) -> TrendCategory:
        """Categorize trend alignment score"""
        
        if trend_score >= 8.0:
            return TrendCategory.EMERGING
        elif trend_score >= 6.0:
            return TrendCategory.MAINSTREAM
        elif trend_score >= 4.0:
            return TrendCategory.CLASSIC
        else:
            return TrendCategory.DECLINING
    
    def _predict_style_longevity(self, description: str, category: str) -> str:
        """Predict how long the style will remain relevant"""
        
        description_lower = description.lower()
        
        # Classic indicators
        classic_keywords = ["classic", "timeless", "traditional", "heritage", "vintage"]
        if any(keyword in description_lower for keyword in classic_keywords):
            return "Long-lasting (5+ years)"
        
        # Trendy indicators
        trendy_keywords = ["trendy", "seasonal", "limited edition", "current", "now"]
        if any(keyword in description_lower for keyword in trendy_keywords):
            return "Short-term (1-2 years)"
        
        # Category-based predictions
        long_lasting_categories = ["jewelry", "watches", "leather goods", "coats"]
        if any(cat in category.lower() for cat in long_lasting_categories):
            return "Medium to long-term (3-5 years)"
        
        return "Medium-term (2-3 years)"
    
    def _estimate_demand_volatility(self, category: str, trend_score: float) -> str:
        """Estimate demand volatility"""
        
        # High volatility categories
        high_volatility = ["clothing", "accessories", "seasonal items"]
        if any(cat in category.lower() for cat in high_volatility):
            if trend_score > 7:
                return "High volatility"
            else:
                return "Medium volatility"
        
        # Low volatility categories
        low_volatility = ["jewelry", "watches", "leather goods"]
        if any(cat in category.lower() for cat in low_volatility):
            return "Low volatility"
        
        return "Medium volatility"
    
    def _identify_target_demographics(self, brand: str, price: float, category: str) -> Dict:
        """Identify target demographics based on product characteristics"""
        
        demographics = {
            "age_range": "25-45",
            "income_level": "Upper-middle class",
            "lifestyle": "Fashion-conscious",
            "gender": "Unisex"
        }
        
        # Adjust based on price
        if price > 5000:
            demographics["age_range"] = "35-55"
            demographics["income_level"] = "High net worth"
            demographics["lifestyle"] = "Luxury lifestyle"
        elif price > 1000:
            demographics["age_range"] = "30-50"
            demographics["income_level"] = "High income"
            demographics["lifestyle"] = "Premium lifestyle"
        elif price < 200:
            demographics["age_range"] = "18-35"
            demographics["income_level"] = "Middle class"
            demographics["lifestyle"] = "Style-conscious"
        
        # Adjust based on category
        category_lower = category.lower()
        if "jewelry" in category_lower or "watch" in category_lower:
            demographics["age_range"] = "30-60"
        elif "casual" in category_lower or "streetwear" in category_lower:
            demographics["age_range"] = "18-30"
        
        return demographics
    
    def _extract_ai_trend_score(self, ai_analysis: str) -> Optional[float]:
        """Extract trend score from AI analysis"""
        try:
            if not ai_analysis:
                return None
            
            # Look for numerical scores in the AI analysis
            import re
            score_patterns = [
                r'trend.*?score.*?(\d+(?:\.\d+)?)',
                r'score.*?(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)/10',
                r'(\d+(?:\.\d+)?)\s*out\s*of\s*10'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, ai_analysis.lower())
                if matches:
                    score = float(matches[0])
                    return min(score, 10.0)
            
            return None
        except Exception as e:
            logger.error(f"Error extracting AI trend score: {e}")
            return None
    
    def _calculate_overall_score(self, trend_score: float, value_score: float, 
                               feature_count: int, price_category: PriceCategory) -> float:
        """Calculate overall product score"""
        try:
            # Weight different factors
            weighted_score = (
                trend_score * 0.3 +
                value_score * 0.3 +
                min(feature_count * 0.5, 3.0) +  # Cap feature bonus at 3 points
                self._get_price_category_bonus(price_category) * 0.2
            )
            
            return min(max(weighted_score, 0.0), 10.0)
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 5.0
    
    def _get_price_category_bonus(self, price_category: PriceCategory) -> float:
        """Get bonus points based on price category positioning"""
        bonuses = {
            PriceCategory.ULTRA_LUXURY: 8.0,
            PriceCategory.LUXURY: 7.0,
            PriceCategory.PREMIUM: 6.0,
            PriceCategory.MID_RANGE: 5.0,
            PriceCategory.BUDGET: 4.0
        }
        return bonuses.get(price_category, 5.0)
    
    def _calculate_confidence_level(self, ai_analysis: str, description: str) -> float:
        """Calculate confidence level of the analysis"""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on available data
            if ai_analysis and len(ai_analysis) > 100:
                confidence += 0.2
            
            if description and len(description) > 50:
                confidence += 0.2
            
            # Check for detailed information
            if ai_analysis and any(word in ai_analysis.lower() for word in 
                                 ['detailed', 'comprehensive', 'thorough']):
                confidence += 0.1
            
            return min(confidence, 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence level: {e}")
            return 0.5
    
    def _generate_customer_personas(self, demographics: Dict[str, str], 
                                  price_category: PriceCategory) -> List[str]:
        """Generate customer personas based on demographics and price"""
        try:
            personas = []
            
            age_range = demographics.get('age_range', '25-45')
            income_level = demographics.get('income_level', 'Middle class')
            lifestyle = demographics.get('lifestyle', 'Fashion-conscious')
            
            # Primary persona
            if price_category in [PriceCategory.LUXURY, PriceCategory.ULTRA_LUXURY]:
                personas.append(f"Affluent professional ({age_range}) with {lifestyle.lower()} lifestyle")
                personas.append("High net worth individual seeking exclusive pieces")
            elif price_category == PriceCategory.PREMIUM:
                personas.append(f"Upwardly mobile professional ({age_range}) with premium taste")
                personas.append("Quality-conscious consumer willing to invest in durable pieces")
            else:
                personas.append(f"Style-conscious individual ({age_range}) seeking good value")
                personas.append("Fashion enthusiast looking for trendy, affordable options")
            
            # Add lifestyle-based personas
            if "luxury" in lifestyle.lower():
                personas.append("Luxury lifestyle enthusiast and brand collector")
            elif "fashion" in lifestyle.lower():
                personas.append("Fashion-forward trendsetter and early adopter")
            
            return personas[:3]  # Return top 3 personas
        except Exception as e:
            logger.error(f"Error generating customer personas: {e}")
            return ["Fashion-conscious consumer", "Quality-seeking buyer"]
    
    def _generate_price_recommendations(self, price: float, category: str, value_score: float) -> List[str]:
        """Generate price optimization recommendations"""
        try:
            recommendations = []
            
            ranges = self.price_ranges.get(category, {"min": 50, "max": 500})
            
            if price < ranges["min"]:
                recommendations.append("Consider increasing price to match category expectations")
            elif price > ranges["max"] * 1.5:
                recommendations.append("Price may be too high for category - justify with premium features")
            
            if value_score < 5.0:
                recommendations.append("Improve value proposition or consider price reduction")
            elif value_score > 8.0:
                recommendations.append("Strong value proposition supports current pricing")
            
            recommendations.append("Monitor competitor pricing for positioning adjustments")
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating price recommendations: {e}")
            return ["Review pricing strategy based on market analysis"]
    
    def _generate_product_improvements(self, missing_features: List[str], 
                                     trend_score: float, value_score: float) -> List[str]:
        """Generate product improvement recommendations"""
        try:
            improvements = []
            
            if missing_features:
                improvements.append(f"Consider adding: {', '.join(missing_features[:3])}")
            
            if trend_score < 5.0:
                improvements.append("Update design to align with current trends")
            
            if value_score < 5.0:
                improvements.append("Enhance features or materials to improve value proposition")
            
            improvements.append("Gather customer feedback for targeted improvements")
            
            if trend_score > 8.0:
                improvements.append("Leverage trend-forward positioning in marketing")
            
            return improvements[:5]
        except Exception as e:
            logger.error(f"Error generating product improvements: {e}")
            return ["Conduct market research for improvement opportunities"]
    
    def _analyze_target_demographics(self, price: float, brand: str, 
                                   category: str, description: str) -> Dict[str, str]:
        """Analyze target demographics - wrapper for _identify_target_demographics"""
        return self._identify_target_demographics(brand, price, category)
    
    def _identify_peak_seasons(self, seasonal_demand: Dict[str, float]) -> List[str]:
        """Identify peak seasons from demand data"""
        try:
            # Sort seasons by demand level
            sorted_seasons = sorted(seasonal_demand.items(), key=lambda x: x[1], reverse=True)
            
            # Return seasons with above-average demand
            avg_demand = sum(seasonal_demand.values()) / len(seasonal_demand)
            peak_seasons = [season for season, demand in sorted_seasons if demand > avg_demand]
            
            return peak_seasons if peak_seasons else [sorted_seasons[0][0]]
        except Exception as e:
            logger.error(f"Error identifying peak seasons: {e}")
            return ["spring"]
    
    def _extract_positioning_from_ai(self, ai_analysis: str) -> str:
        """Extract market positioning insights from AI analysis"""
        try:
            if not ai_analysis:
                return ""
            
            # Look for positioning-related keywords and phrases
            positioning_keywords = [
                "positioned as", "market position", "positioning", "targets",
                "appeals to", "designed for", "caters to", "aimed at"
            ]
            
            sentences = ai_analysis.split('.')
            positioning_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in positioning_keywords):
                    positioning_sentences.append(sentence)
            
            if positioning_sentences:
                # Return the most relevant positioning statement
                return positioning_sentences[0].strip()
            
            # If no explicit positioning found, try to extract from context
            if "luxury" in ai_analysis.lower():
                return "Positioned in the luxury segment"
            elif "premium" in ai_analysis.lower():
                return "Positioned as a premium offering"
            elif "affordable" in ai_analysis.lower() or "budget" in ai_analysis.lower():
                return "Positioned as an affordable option"
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting positioning from AI analysis: {e}")
            return ""
    
    def get_analysis_summary(self, insight: ProductInsight) -> Dict:
        """Get a summary of the analysis results"""
        return {
            "product": f"{insight.brand} {insight.product_name}",
            "overall_score": insight.overall_score,
            "trend_score": insight.trend_alignment_score,
            "value_score": insight.value_score,
            "market_position": insight.market_positioning,
            "key_recommendations": insight.marketing_recommendations[:3],
            "confidence": insight.confidence_level
        }
