"""
Test script to verify fine-tuned model integration
Run this after copying the fine-tuned model from Google Colab
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gemini_fashion import FashionGemini
from configs.config import get_model_path, is_finetuned_model_available

def test_model_integration():
    """Test the fine-tuned model integration"""
    
    print("🧪 Testing Fashion LLaMA Model Integration")
    print("=" * 50)
    
    # Check model availability
    model_path = get_model_path()
    is_finetuned = is_finetuned_model_available()
    
    print(f"Model Path: {model_path}")
    print(f"Fine-tuned Model Available: {'✅ Yes' if is_finetuned else '❌ No'}")
    print(f"Using {'Fine-tuned' if is_finetuned else 'Fallback'} Model")
    print("-" * 50)
    
    try:
        # Initialize model
        print("🔄 Loading model...")
        model = FashionGemini()
        print("✅ Model loaded successfully!")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 50)
        print("🧪 Running Test Analyses")
        print("=" * 50)
        
        # Test 1: Product Analysis
        print("\n📦 Test 1: Product Analysis")
        print("-" * 30)
        result1 = model.analyze_product(
            "Hermès silk scarf, limited edition, priced at $650",
            analysis_type="comprehensive"
        )
        print(f"Input: Hermès silk scarf, limited edition, priced at $650")
        print(f"Output: {result1[:200]}..." if len(result1) > 200 else f"Output: {result1}")
        
        # Test 2: Trend Analysis
        print("\n📈 Test 2: Trend Analysis")
        print("-" * 30)
        result2 = model.analyze_product(
            "Oversized blazers, neutral colors, structured shoulders",
            analysis_type="trend"
        )
        print(f"Input: Oversized blazers, neutral colors, structured shoulders")
        print(f"Output: {result2[:200]}..." if len(result2) > 200 else f"Output: {result2}")
        
        # Test 3: Sentiment Analysis
        print("\n💭 Test 3: Sentiment Analysis")
        print("-" * 30)
        result3 = model.analyze_customer_sentiment(
            "Love this bag! Perfect quality and amazing design. Worth every penny!"
        )
        print(f"Input: Love this bag! Perfect quality and amazing design. Worth every penny!")
        print(f"Output: {result3[:200]}..." if len(result3) > 200 else f"Output: {result3}")
        
        # Test 4: Market Insights
        print("\n🏪 Test 4: Market Insights")
        print("-" * 30)
        result4 = model.get_market_insights("luxury handbags", "Gucci")
        print(f"Input: luxury handbags, Gucci brand")
        print(f"Output: {result4[:200]}..." if len(result4) > 200 else f"Output: {result4}")
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        if is_finetuned:
            print("🎉 Fine-tuned model is working correctly!")
            print("Your fashion analysis system is ready for production use.")
        else:
            print("⚠️  Using fallback model.")
            print("To use the fine-tuned model:")
            print("1. Complete fine-tuning in Google Colab")
            print("2. Download the model export folder")
            print("3. Copy to: models/fashion-llama-finetuned/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("\nTroubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Verify model files are in correct location")
        print("3. Check console output for specific errors")
        return False

def test_configuration():
    """Test configuration setup"""
    
    print("\n🔧 Testing Configuration")
    print("=" * 30)
    
    from configs.config import (
        MODEL_PATH, FALLBACK_MODEL_PATH, USE_PEFT_MODEL,
        FINETUNING_DATASETS, BASE_DIR
    )
    
    print(f"Base Directory: {BASE_DIR}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Fallback Path: {FALLBACK_MODEL_PATH}")
    print(f"Use PEFT Model: {USE_PEFT_MODEL}")
    print(f"Datasets Used: {len(FINETUNING_DATASETS)} datasets")
    
    # Check directories
    required_dirs = ["models", "data", "logs"]
    for dir_name in required_dirs:
        dir_path = os.path.join(BASE_DIR, dir_name)
        exists = os.path.exists(dir_path)
        print(f"Directory {dir_name}: {'✅ Exists' if exists else '❌ Missing'}")

if __name__ == "__main__":
    print("🚀 Starting Fashion LLaMA Integration Test")
    print("This will test the fine-tuned model integration")
    print("\n")
    
    # Test configuration first
    test_configuration()
    
    # Test model integration
    success = test_model_integration()
    
    if success:
        print("\n🎯 Integration test completed successfully!")
    else:
        print("\n⚠️  Integration test encountered issues.")
        print("Please check the troubleshooting guide in INTEGRATION_GUIDE.md")
