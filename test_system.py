"""
Simple test script to verify the AI Luxury Brand system is working
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if environment is properly configured"""
    print("🔧 Testing Environment Configuration...")
    
    # Check environment variables
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print(f"✅ Google API Key configured: {api_key[:10]}...")
    else:
        print("❌ Google API Key not found")
        return False
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n📦 Testing Module Imports...")
    
    try:
        from src.utils.sentiment_analyzer import SentimentAnalyzer
        print("✅ SentimentAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ SentimentAnalyzer import failed: {e}")
        return False
    
    try:
        from src.utils.text_processing import TextProcessor
        print("✅ TextProcessor imported successfully")
    except Exception as e:
        print(f"❌ TextProcessor import failed: {e}")
        return False
    
    try:
        from src.analysis.product_analyzer import ProductAnalyzer
        print("✅ ProductAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ ProductAnalyzer import failed: {e}")
        return False
    
    try:
        from src.agents.market_agent import MarketResearchAgent
        print("✅ MarketResearchAgent imported successfully")
    except Exception as e:
        print(f"❌ MarketResearchAgent import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without heavy model loading"""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        from src.utils.text_processing import TextProcessor
        
        processor = TextProcessor()
        
        # Test keyword extraction
        sample_text = "luxury designer handbag made of genuine leather"
        keywords = processor.extract_keywords(sample_text)
        print(f"✅ Keyword extraction working: {keywords}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Luxury Brand System Test\n")
    
    # Run tests
    env_ok = test_environment()
    imports_ok = test_imports()
    basic_ok = test_basic_functionality()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Basic Functions: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    
    if all([env_ok, imports_ok, basic_ok]):
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("\n📋 Next steps:")
        print("1. Start the FastAPI server: uvicorn src.api.main:app --reload")
        print("2. Start the Streamlit app: streamlit run src/ui/streamlit_app.py")
        print("3. Open http://localhost:8000/docs for API documentation")
        print("4. Open http://localhost:8501 for the web interface")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
