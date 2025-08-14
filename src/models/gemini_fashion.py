"""
Gemini AI integration for fashion analysis
"""
import google.generativeai as genai
import os
from typing import Dict, List, Optional
from loguru import logger
import json
from datetime import datetime

class FashionGemini:
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini AI model for fashion analysis
        
        Args:
            api_key: Google AI API key for Gemini
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generation config for consistent responses
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        
        logger.info("Gemini AI model initialized successfully")
    
    def analyze_product(self, product_description: str, analysis_type: str = "comprehensive") -> str:
        """
        Analyze a product using Gemini AI
        
        Args:
            product_description: Description of the fashion product
            analysis_type: Type of analysis (comprehensive, trend, pricing, etc.)
        
        Returns:
            Analysis results as text
        """
        prompt = self._create_analysis_prompt(product_description, analysis_type)
        
        try:
            # Generate analysis using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}")
            return f"Error analyzing product: {str(e)}"
    
    def _create_analysis_prompt(self, product_description: str, analysis_type: str) -> str:
        """Create appropriate prompt for the analysis"""
        
        base_prompt = f"""
You are a luxury fashion expert AI with deep knowledge of the fashion industry, market trends, and consumer behavior. 
Analyze the following product description and provide detailed insights:

Product Description: {product_description}

Analysis Type: {analysis_type}

Please provide a comprehensive analysis including:
"""
        
        if analysis_type == "comprehensive":
            prompt = base_prompt + """
1. **Market Positioning Analysis**: Where this product fits in the competitive landscape (luxury/premium/mid-tier/accessible)
2. **Trend Alignment Score**: How well it matches current/emerging trends (score 1-10 with detailed explanation)
3. **Price-Value Assessment**: Competitive pricing analysis with specific recommendations
4. **Feature Gap Analysis**: Missing features compared to successful competitors
5. **Target Audience**: Ideal customer demographics and personas (age, income, lifestyle)
6. **Seasonal Demand Forecast**: Expected demand patterns throughout the year
7. **Competitive Landscape**: Main competitors and differentiation opportunities
8. **Marketing Strategy**: Recommended positioning and messaging
9. **Brand Alignment**: How product fits with brand identity

Provide specific, actionable insights for each category. Use data-driven reasoning and industry expertise.
Format as structured analysis with clear headings and bullet points.
"""
        
        elif analysis_type == "trend":
            prompt = base_prompt + """
1. **Current Trend Alignment**: Score 1-10 with detailed explanation
2. **Emerging Trend Potential**: How well positioned for upcoming trends
3. **Fashion Cycle Position**: Where the product sits in the fashion lifecycle
4. **Style Longevity Prediction**: Expected lifespan and evolution
5. **Trend Recommendations**: Specific improvements to increase trend relevance

Focus on current 2024-2025 fashion trends and provide specific recommendations.
"""
        
        elif analysis_type == "pricing":
            prompt = base_prompt + """
1. **Price Positioning**: Analysis of current price point in the market
2. **Value Proposition Analysis**: What customers get for the price
3. **Competitive Pricing Comparison**: How it compares to similar products
4. **Price Optimization Recommendations**: Specific pricing strategies
5. **Price Elasticity Insights**: Expected demand sensitivity to price changes

Provide specific price recommendations and justifications.
"""
        
        else:
            prompt = base_prompt + "\nProvide detailed analysis based on the product description."
        
        return prompt
    
    def get_market_insights(self, category: str, brand: str = None) -> str:
        """Get market insights for a specific category or brand"""
        
        prompt = f"""
As a luxury fashion market analyst, provide comprehensive insights about:

Category: {category}
{f"Brand: {brand}" if brand else "Focus on overall category analysis"}

Please analyze:
1. **Market Size and Growth**: Current market size and growth projections
2. **Key Players and Market Share**: Leading brands and their positioning
3. **Consumer Preferences**: Current consumer behavior and preferences
4. **Emerging Opportunities**: New market opportunities and niches
5. **Challenges and Threats**: Market challenges and competitive threats
6. **Future Outlook**: 2024-2025 predictions and trends

Provide specific, data-driven insights with actionable recommendations.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating market insights: {e}")
            return f"Error generating insights: {str(e)}"
    
    def predict_trends(self, timeframe: str = "next_season") -> str:
        """Predict fashion trends for specified timeframe"""
        
        prompt = f"""
As a fashion trend forecasting expert, predict the key trends for {timeframe}:

Consider current global fashion movements, cultural shifts, and consumer behavior changes.

Please provide predictions for:
1. **Color Trends**: Dominant and emerging color palettes
2. **Style and Silhouette Trends**: Key shapes, cuts, and design elements
3. **Material and Fabric Trends**: Popular and emerging materials
4. **Sustainable Fashion Trends**: Eco-conscious and ethical fashion movements
5. **Technology Integration Trends**: Smart textiles and tech-enabled fashion
6. **Consumer Behavior Trends**: How shopping and wearing patterns are evolving

Provide specific, actionable trend predictions with confidence levels and reasoning.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error predicting trends: {e}")
            return f"Error predicting trends: {str(e)}"
    
    def analyze_competitor_landscape(self, product_description: str) -> str:
        """Analyze competitive landscape for a product"""
        
        prompt = f"""
As a competitive intelligence expert in luxury fashion, analyze the competitive landscape for:

Product: {product_description}

Provide analysis on:
1. **Direct Competitors**: Top 5 brands/products that compete directly
2. **Indirect Competitors**: Alternative products that serve similar needs
3. **Competitive Advantages**: What makes each competitor strong
4. **Market Gaps**: Opportunities not being addressed by competitors
5. **Competitive Threats**: Potential risks from competitor actions
6. **Positioning Strategy**: How to position against competitors

Include specific brand names, price points, and strategic recommendations.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing competitors: {e}")
            return f"Error analyzing competitors: {str(e)}"
    
    def generate_marketing_strategy(self, product_description: str, target_audience: str) -> str:
        """Generate marketing strategy recommendations"""
        
        prompt = f"""
As a luxury fashion marketing strategist, develop a comprehensive marketing strategy for:

Product: {product_description}
Target Audience: {target_audience}

Create strategy covering:
1. **Brand Positioning**: How to position the product in the market
2. **Key Messaging**: Core messages that will resonate with target audience
3. **Channel Strategy**: Best marketing channels and platforms
4. **Content Strategy**: Types of content that will drive engagement
5. **Influencer Strategy**: Types of influencers and partnerships
6. **Seasonal Campaigns**: Year-round marketing calendar
7. **Budget Allocation**: Recommended budget distribution across channels

Provide specific, actionable recommendations with reasoning.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating marketing strategy: {e}")
            return f"Error generating marketing strategy: {str(e)}"
    
    def extract_structured_insights(self, product_description: str) -> Dict:
        """Extract structured insights that can be used programmatically"""
        
        prompt = f"""
Analyze this fashion product and return structured insights in JSON format:

Product: {product_description}

Return a JSON object with the following structure:
{{
    "market_positioning": {{
        "segment": "luxury/premium/mid-market/accessible",
        "target_age_group": "age range",
        "price_tier": "price category",
        "positioning_statement": "brief positioning"
    }},
    "trend_analysis": {{
        "trend_alignment_score": "score out of 10",
        "aligned_trends": ["list", "of", "trends"],
        "trend_risks": ["potential", "risks"],
        "style_longevity": "high/medium/low"
    }},
    "competitive_analysis": {{
        "direct_competitors": ["brand1", "brand2", "brand3"],
        "competitive_advantages": ["advantage1", "advantage2"],
        "market_gaps": ["gap1", "gap2"]
    }},
    "pricing_insights": {{
        "price_positioning": "expensive/competitive/affordable",
        "value_proposition": "key value points",
        "pricing_recommendations": ["rec1", "rec2"]
    }},
    "seasonal_demand": {{
        "spring": 0.25,
        "summer": 0.25,
        "fall": 0.25,
        "winter": 0.25
    }}
}}

Ensure the JSON is valid and complete.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Try to parse JSON response
            try:
                structured_data = json.loads(response.text)
                return structured_data
            except json.JSONDecodeError:
                # If JSON parsing fails, return a structured error response
                logger.warning("Could not parse JSON response from Gemini")
                return {
                    "error": "Could not parse structured response",
                    "raw_response": response.text
                }
            
        except Exception as e:
            logger.error(f"Error extracting structured insights: {e}")
            return {
                "error": f"Error extracting insights: {str(e)}"
            }
    
    def analyze_product_comprehensive(self, product_data: Dict) -> Dict:
        """
        Comprehensive product analysis returning structured data
        
        Args:
            product_data: Dictionary containing product information
                - name: Product name
                - brand: Brand name
                - price: Price (numeric)
                - description: Product description
                - category: Product category (optional)
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        
        # Build product description string
        product_description = f"""
Product Name: {product_data.get('name', '')}
Brand: {product_data.get('brand', '')}
Price: ${product_data.get('price', 0)}
Description: {product_data.get('description', '')}
Category: {product_data.get('category', 'Fashion')}
"""
        
        prompt = f"""
You are a luxury fashion expert AI. Analyze this product and provide comprehensive insights in JSON format:

{product_description}

Return a JSON object with the following structure:
{{
    "market_positioning": {{
        "segment": "luxury/premium/mid-market/accessible",
        "target_demographics": {{
            "age_range": "age range",
            "income_level": "income bracket",
            "lifestyle": "lifestyle description"
        }},
        "positioning_score": "score out of 10",
        "positioning_statement": "brief positioning statement"
    }},
    "trend_alignment": {{
        "trend_score": "score out of 10",
        "aligned_trends": ["current trend 1", "current trend 2"],
        "emerging_trends": ["emerging trend 1", "emerging trend 2"],
        "trend_risks": ["potential risk 1", "potential risk 2"],
        "style_longevity": "classic/trendy/fad"
    }},
    "price_value_assessment": {{
        "price_tier": "budget/mid-range/premium/luxury/ultra-luxury",
        "value_score": "score out of 10",
        "competitive_pricing": "below/at/above market",
        "price_recommendations": ["recommendation 1", "recommendation 2"],
        "value_drivers": ["quality", "brand", "exclusivity"]
    }},
    "feature_gap_analysis": {{
        "missing_features": ["feature 1", "feature 2"],
        "improvement_opportunities": ["opportunity 1", "opportunity 2"],
        "innovation_potential": "high/medium/low"
    }},
    "seasonal_demand": {{
        "spring": 0.25,
        "summer": 0.25,
        "fall": 0.25,
        "winter": 0.25,
        "peak_season": "season name",
        "demand_drivers": ["driver 1", "driver 2"]
    }},
    "competitive_landscape": {{
        "direct_competitors": ["competitor 1", "competitor 2", "competitor 3"],
        "competitive_advantages": ["advantage 1", "advantage 2"],
        "threats": ["threat 1", "threat 2"],
        "market_share_potential": "high/medium/low"
    }},
    "marketing_strategy": {{
        "key_messages": ["message 1", "message 2"],
        "channels": ["channel 1", "channel 2"],
        "influencer_potential": "high/medium/low",
        "content_themes": ["theme 1", "theme 2"]
    }},
    "overall_assessment": {{
        "market_potential": "high/medium/low",
        "investment_rating": "strong buy/buy/hold/avoid",
        "risk_level": "low/medium/high",
        "success_probability": "score out of 10"
    }}
}}

Ensure the JSON is valid and complete. Base your analysis on current fashion industry trends and market data.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Try to parse JSON response
            try:
                # Clean up the response to extract JSON
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove ```
                
                structured_data = json.loads(response_text)
                
                # Add metadata
                structured_data['metadata'] = {
                    'analysis_date': datetime.now().isoformat(),
                    'model_version': 'gemini-1.5-flash',
                    'product_analyzed': product_data.get('name', 'Unknown Product')
                }
                
                return structured_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON response from Gemini: {e}")
                # Return fallback structured response
                return self._create_fallback_analysis(product_data, response.text)
            
        except Exception as e:
            logger.error(f"Error during comprehensive analysis: {e}")
            return self._create_error_response(str(e))
    
    def _create_fallback_analysis(self, product_data: Dict, raw_response: str) -> Dict:
        """Create fallback structured analysis when JSON parsing fails"""
        return {
            "market_positioning": {
                "segment": "mid-market",
                "target_demographics": {
                    "age_range": "25-45",
                    "income_level": "middle income",
                    "lifestyle": "fashion-conscious"
                },
                "positioning_score": "6",
                "positioning_statement": "Moderate market positioning"
            },
            "trend_alignment": {
                "trend_score": "6",
                "aligned_trends": ["classic style"],
                "emerging_trends": ["sustainable fashion"],
                "trend_risks": ["fast fashion competition"],
                "style_longevity": "trendy"
            },
            "error": "Could not parse structured response",
            "raw_analysis": raw_response,
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "model_version": "gemini-1.5-flash",
                "product_analyzed": product_data.get('name', 'Unknown Product'),
                "status": "fallback_analysis"
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response structure"""
        return {
            "error": True,
            "error_message": error_message,
            "analysis_date": datetime.now().isoformat(),
            "status": "analysis_failed"
        }
