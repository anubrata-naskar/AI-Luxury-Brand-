"""
Streamlit web interface for the AI Fashion Analysis System
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="AI Luxury Fashion Analysis",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def call_api(endpoint: str, method: str = "GET", data: dict = None):
    """Call the FastAPI backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("🚨 Cannot connect to the API server. Please ensure the FastAPI server is running on localhost:8000")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 API request failed: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">👗 AI Luxury Fashion Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Product Analysis", "Market Research", "Trend Analysis", "Dashboard"]
    )
    
    # Check API health
    health_status = call_api("/health")
    if health_status:
        if health_status.get("status") == "healthy":
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.warning("⚠️ API Issues Detected")
    else:
        st.sidebar.error("❌ API Disconnected")
    
    # Route to different pages
    if page == "Product Analysis":
        product_analysis_page()
    elif page == "Market Research":
        market_research_page()
    elif page == "Trend Analysis":
        trend_analysis_page()
    elif page == "Dashboard":
        dashboard_page()

def product_analysis_page():
    """Product Analysis Interface"""
    st.header("🔍 Product Analysis")
    st.write("Analyze luxury fashion products for market insights and recommendations")
    
    # Input form
    with st.form("product_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input("Product Name", placeholder="e.g., Cashmere Sweater")
            brand = st.text_input("Brand", placeholder="e.g., Gucci")
            price = st.number_input("Price ($)", min_value=0.0, step=10.0, value=100.0)
        
        with col2:
            categories = ["dresses", "sweaters", "jeans", "shoes", "bags", "accessories", "coats", "jackets", "blouses", "skirts", "pants", "jewelry"]
            category = st.selectbox("Category", categories)
            
        description = st.text_area(
            "Product Description",
            placeholder="Describe the product features, materials, style, etc.",
            height=100
        )
        
        submitted = st.form_submit_button("🚀 Analyze Product", use_container_width=True)
    
    if submitted and product_name and brand and description:
        with st.spinner("Analyzing product... This may take a few moments."):
            # Prepare API request
            analysis_request = {
                "name": product_name,
                "brand": brand,
                "price": price,
                "description": description,
                "category": category
            }
            
            # Call API
            result = call_api("/analyze/product", method="POST", data=analysis_request)
            
            if result and result.get("status") == "success":
                display_product_analysis(result["data"])
            elif result:
                st.error(f"Analysis failed: {result.get('detail', 'Unknown error')}")

def display_product_analysis(data):
    """Display product analysis results"""
    st.success("✅ Analysis Complete!")
    
    # Overview metrics
    st.subheader("📊 Analysis Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = data.get('overall_score', 0)
        st.metric("Overall Score", f"{overall_score:.1f}/10" if overall_score is not None else "0.0/10")
    with col2:
        trend_alignment = data.get('trend_alignment_score', 0)
        st.metric("Trend Alignment", f"{trend_alignment:.1f}/10" if trend_alignment is not None else "0.0/10")
    with col3:
        value_score = data.get('value_score', 0)
        st.metric("Value Score", f"{value_score:.1f}/10" if value_score is not None else "0.0/10")
    with col4:
        confidence = data.get('confidence_level', 0)
        st.metric("Confidence", f"{confidence*100:.0f}%" if confidence is not None else "0%")
    
    # Detailed analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["Market Position", "Features", "Target Audience", "Recommendations"])
    
    with tab1:
        st.subheader("🎯 Market Positioning")
        st.write(data.get('market_positioning', 'No positioning analysis available'))
        
        st.subheader("🏆 Competitive Landscape")
        competitors = data.get('competitive_landscape', [])
        if competitors:
            for i, competitor in enumerate(competitors[:5], 1):
                st.write(f"{i}. {competitor}")
        
        st.subheader("⭐ Unique Selling Points")
        usps = data.get('unique_selling_points', [])
        if usps:
            for usp in usps:
                st.write(f"• {usp}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Key Features")
            features = data.get('key_features', [])
            if features:
                for feature in features:
                    st.write(f"• {feature}")
            else:
                st.write("No key features identified")
        
        with col2:
            st.subheader("❌ Missing Features")
            missing = data.get('missing_features', [])
            if missing:
                for feature in missing:
                    st.write(f"• {feature}")
            else:
                st.write("No missing features identified")
    
    with tab3:
        st.subheader("👥 Target Demographics")
        demographics = data.get('target_demographics', {})
        if demographics:
            for key, value in demographics.items():
                st.write(f"**{key.title()}:** {value}")
        
        st.subheader("🎭 Customer Personas")
        personas = data.get('customer_personas', [])
        if personas:
            for i, persona in enumerate(personas, 1):
                st.write(f"{i}. {persona}")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Marketing Recommendations")
            marketing = data.get('marketing_recommendations', [])
            if marketing:
                for rec in marketing:
                    st.write(f"• {rec}")
        
        with col2:
            st.subheader("🔧 Product Improvements")
            improvements = data.get('product_improvements', [])
            if improvements:
                for imp in improvements:
                    st.write(f"• {imp}")
    
    # Seasonal demand chart
    seasonal_data = data.get('seasonal_demand', {})
    if seasonal_data:
        st.subheader("📅 Seasonal Demand Forecast")
        df_seasonal = pd.DataFrame(list(seasonal_data.items()), columns=['Season', 'Demand'])
        fig = px.bar(df_seasonal, x='Season', y='Demand', title="Expected Seasonal Demand")
        st.plotly_chart(fig, use_container_width=True)

def market_research_page():
    """Market Research Interface"""
    st.header("🔬 Market Research")
    st.write("Research competitors and analyze market trends")
    
    with st.form("market_research_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_description = st.text_input(
                "Product/Search Query",
                placeholder="e.g., luxury cashmere sweaters"
            )
            category = st.selectbox(
                "Category",
                ["dresses", "sweaters", "jeans", "shoes", "bags", "accessories", "coats", "jackets", "blouses", "skirts", "pants", "jewelry"]
            )
        
        with col2:
            max_products = st.slider("Max Products to Analyze", 10, 100, 50)
            col_a, col_b = st.columns(2)
            with col_a:
                min_price = st.number_input("Min Price ($)", min_value=0.0, value=0.0)
            with col_b:
                max_price = st.number_input("Max Price ($)", min_value=0.0, value=1000.0)
        
        submitted = st.form_submit_button("🕵️ Start Research", use_container_width=True)
    
    if submitted and product_description:
        with st.spinner("Researching market... This may take several minutes."):
            research_request = {
                "product_description": product_description,
                "category": category,
                "max_products": max_products,
                "min_price": min_price,
                "max_price": max_price
            }
            
            result = call_api("/research/market", method="POST", data=research_request)
            
            if result and result.get("status") == "success":
                display_market_research(result["data"])
            elif result:
                st.error(f"Research failed: {result.get('detail', 'Unknown error')}")

def display_market_research(data):
    """Display market research results"""
    st.success("✅ Market Research Complete!")
    
    # Overview metrics
    st.subheader("📊 Research Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Products Found", data.get('total_products_found', 0))
    with col2:
        price_analysis = data.get('price_analysis', {})
        avg_price = price_analysis.get('avg_price', 0)
        st.metric("Average Price", f"${avg_price:.0f}" if avg_price is not None else "$0")
    with col3:
        sentiment = data.get('sentiment_analysis', {})
        overall_sentiment = sentiment.get('overall_sentiment', 0)
        st.metric("Market Sentiment", f"{overall_sentiment:.2f}" if overall_sentiment is not None else "0.00")
    with col4:
        avg_rating = data.get('average_rating', 0)
        st.metric("Average Rating", f"{avg_rating:.1f}⭐" if avg_rating is not None else "0.0⭐")
    
    # Detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Top Brands", "Market Trends", "Products"])
    
    with tab1:
        st.subheader("💰 Price Analysis")
        price_data = data.get('price_analysis', {})
        if price_data:
            col1, col2 = st.columns(2)
            with col1:
                min_price = price_data.get('min_price', 0)
                max_price = price_data.get('max_price', 0)
                st.metric("Min Price", f"${min_price:.0f}" if min_price is not None else "$0")
                st.metric("Max Price", f"${max_price:.0f}" if max_price is not None else "$0")
            with col2:
                median_price = price_data.get('median_price', 0)
                price_std = price_data.get('price_std', 0)
                st.metric("Median Price", f"${median_price:.0f}" if median_price is not None else "$0")
                st.metric("Price Std Dev", f"${price_std:.0f}" if price_std is not None else "$0")
    
    with tab2:
        st.subheader("🏆 Top Brands")
        top_brands = data.get('top_brands', [])
        if top_brands:
            for i, brand in enumerate(top_brands[:10], 1):
                st.write(f"{i}. {brand}")
    
    with tab3:
        st.subheader("📈 Market Trends")
        trends = data.get('market_trends', [])
        if trends:
            for trend in trends:
                st.write(f"• {trend}")
    
    with tab4:
        st.subheader("🛍️ Sample Products")
        products = data.get('competitor_products', [])
        if products:
            df_products = pd.DataFrame(products[:20])
            st.dataframe(df_products[['name', 'brand', 'price', 'average_rating']], use_container_width=True)

def trend_analysis_page():
    """Trend Analysis Interface"""
    st.header("📈 Trend Analysis")
    st.write("Analyze fashion trends and get predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔮 Predict Next Season Trends", use_container_width=True):
            with st.spinner("Analyzing trends..."):
                result = call_api("/analyze/trends", method="POST", data={"category": "general", "timeframe": "next_season"})
                if result and result.get("status") == "success":
                    st.success("✅ Trend Analysis Complete!")
                    st.write(result["data"]["trend_predictions"])
    
    with col2:
        if st.button("📊 Get Market Insights", use_container_width=True):
            with st.spinner("Gathering insights..."):
                result = call_api("/analyze/market-insights", method="POST", data={"category": "luxury_fashion"})
                if result and result.get("status") == "success":
                    st.success("✅ Market Insights Ready!")
                    st.write(result["data"]["market_insights"])

def dashboard_page():
    """Dashboard with overview metrics"""
    st.header("📊 Dashboard")
    st.write("Overview of system performance and analytics")
    
    # Mock data for demonstration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", "1,234", "+12%")
    with col2:
        st.metric("Products Analyzed", "856", "+8%")
    with col3:
        st.metric("Market Research", "45", "+15%")
    with col4:
        st.metric("API Uptime", "99.9%", "+0.1%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample trend data
        trend_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Analyses': [20 + i + (i % 7) * 5 for i in range(30)]
        })
        fig = px.line(trend_data, x='Date', y='Analyses', title="Daily Analysis Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sample category distribution
        category_data = pd.DataFrame({
            'Category': ['Dresses', 'Sweaters', 'Shoes', 'Bags', 'Accessories'],
            'Count': [45, 38, 32, 28, 22]
        })
        fig = px.pie(category_data, values='Count', names='Category', title="Analysis by Category")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
