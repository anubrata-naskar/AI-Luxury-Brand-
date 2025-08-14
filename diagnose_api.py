"""
Diagnostic script to identify API connection issues
"""
import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_internet_connection():
    """Test basic internet connectivity"""
    print("🌐 Testing Internet Connection...")
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connection OK")
            return True
        else:
            print(f"❌ Internet connection issue: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Internet connection failed: {e}")
        return False

def test_google_api_key():
    """Test Google API key validity"""
    print("\n🔑 Testing Google API Key...")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API key with a simple request
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try to list models to test API connectivity
        models = genai.list_models()
        model_list = list(models)
        print(f"✅ Google API connection successful! Found {len(model_list)} models")
        return True
        
    except Exception as e:
        print(f"❌ Google API connection failed: {e}")
        return False

def test_firewall_and_proxy():
    """Test for firewall or proxy issues"""
    print("\n🛡️ Testing Firewall/Proxy Issues...")
    
    # Test specific Google AI endpoints
    endpoints_to_test = [
        "https://generativelanguage.googleapis.com",
        "https://ai.google.dev",
        "https://googleapis.com"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"✅ {endpoint} - Status: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"⏱️ {endpoint} - Timeout (possible firewall block)")
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint} - Connection error (possible firewall block)")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")

def test_environment_file():
    """Test .env file configuration"""
    print("\n📁 Testing Environment File...")
    
    env_path = ".env"
    if os.path.exists(env_path):
        print("✅ .env file exists")
        
        # Read and check contents
        with open(env_path, 'r') as f:
            content = f.read()
            
        if "GOOGLE_API_KEY" in content:
            print("✅ GOOGLE_API_KEY found in .env file")
        else:
            print("❌ GOOGLE_API_KEY not found in .env file")
            
        if "AIzaSy" in content:
            print("✅ API key format looks correct")
        else:
            print("❌ API key format may be incorrect")
            
    else:
        print("❌ .env file not found")
        return False
    
    return True

def test_simple_api_call():
    """Test a simple API call to Google Gemini"""
    print("\n🤖 Testing Simple API Call...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        
        # Try to create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test prompt
        response = model.generate_content("Hello, please respond with 'API Working'")
        
        print(f"✅ API Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 AI Luxury Brand API Diagnostics\n")
    
    # Run tests
    internet_ok = test_internet_connection()
    env_ok = test_environment_file()
    firewall_ok = test_firewall_and_proxy()
    api_key_ok = test_google_api_key()
    
    if all([internet_ok, env_ok, api_key_ok]):
        api_call_ok = test_simple_api_call()
    else:
        api_call_ok = False
    
    # Summary
    print("\n📊 Diagnostic Summary:")
    print(f"Internet Connection: {'✅ PASS' if internet_ok else '❌ FAIL'}")
    print(f"Environment File: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"Google API Key: {'✅ PASS' if api_key_ok else '❌ FAIL'}")
    print(f"API Call Test: {'✅ PASS' if api_call_ok else '❌ FAIL'}")
    
    if all([internet_ok, env_ok, api_key_ok, api_call_ok]):
        print("\n🎉 All API tests passed! Your connection is working.")
    else:
        print("\n🔧 Issues found. Here's how to fix them:")
        
        if not internet_ok:
            print("1. Check your internet connection")
            
        if not env_ok:
            print("2. Make sure .env file exists with GOOGLE_API_KEY")
            
        if not api_key_ok:
            print("3. Verify your Google API key is valid:")
            print("   - Go to https://aistudio.google.com/app/apikey")
            print("   - Generate a new API key if needed")
            print("   - Update your .env file")
            
        if not api_call_ok:
            print("4. Try restarting your application")

if __name__ == "__main__":
    main()
