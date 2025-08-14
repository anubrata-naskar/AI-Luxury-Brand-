"""
Configuration settings for AI Luxury Fashion Brand Analysis System
"""

import os
from typing import Dict, Any

# Base configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Configuration (Updated for fine-tuned model integration)
MODEL_NAME = "fashion-llama-3b-finetuned"
MODEL_PATH = os.path.join(BASE_DIR, "models", "fashion-llama-finetuned")
FALLBACK_MODEL_PATH = "microsoft/DialoGPT-medium"

# Fine-tuning configuration
USE_PEFT_MODEL = True  # True for LoRA fine-tuned models from Colab
USE_4BIT_QUANTIZATION = True
DEVICE_MAP = "auto"
TORCH_DTYPE = "float16"

# Generation parameters
MAX_LENGTH = 2048
TEMPERATURE = 0.7
TOP_P = 0.95
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 512

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG = True
LOG_LEVEL = "info"

# Database Configuration
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'fashion_analysis.db')}"

# Scraping Configuration
SCRAPING_DELAY = 1
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
MAX_CONCURRENT_REQUESTS = 10
TIMEOUT = 30

# Fine-tuning datasets used (for reference)
FINETUNING_DATASETS = [
    "Fashion Product Feedback Dataset (OpenDataBay, 2025)",
    "Amazon Fashion Reviews (Amazon Product Data subset)", 
    "Data4Fashion Text Dataset",
    "Myntra Product Data (Kaggle)",
    "H&M Product Descriptions (Kaggle)"
]

# Analysis Configuration
SENTIMENT_THRESHOLD = 0.5
TREND_CONFIDENCE_THRESHOLD = 0.7
PRICE_ANALYSIS_RANGE = 0.2

def get_model_path() -> str:
    """
    Get the model path, checking if fine-tuned model exists
    Returns fallback model if fine-tuned model not found
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json")):
        print(f"Using fine-tuned model: {MODEL_PATH}")
        return MODEL_PATH
    else:
        print(f"Fine-tuned model not found at {MODEL_PATH}")
        print(f"Using fallback model: {FALLBACK_MODEL_PATH}")
        print("To use the fine-tuned model:")
        print("1. Complete fine-tuning in Google Colab")
        print("2. Download the model export folder") 
        print(f"3. Copy to: {MODEL_PATH}")
        return FALLBACK_MODEL_PATH

def is_finetuned_model_available() -> bool:
    """Check if the fine-tuned model is available"""
    adapter_config = os.path.join(MODEL_PATH, "adapter_config.json")
    return os.path.exists(adapter_config)

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# API Keys (set in .env file)
OPENAI_API_KEY = ""
SERP_API_KEY = ""
RAPIDAPI_KEY = ""

# Analysis Parameters
TREND_WEIGHT = 0.3
PRICE_WEIGHT = 0.25
FEATURE_WEIGHT = 0.25
SENTIMENT_WEIGHT = 0.2

# Supported Categories
FASHION_CATEGORIES = [
    "dresses", "sweaters", "jeans", "shoes", "bags", "accessories",
    "coats", "jackets", "blouses", "skirts", "pants", "jewelry"
]

# Luxury Brands List
LUXURY_BRANDS = [
    "Gucci", "Louis Vuitton", "Chanel", "Hermès", "Prada", "Dior",
    "Burberry", "Versace", "Saint Laurent", "Balenciaga", "Bottega Veneta",
    "Fendi", "Givenchy", "Valentino", "Tom Ford", "Alexander McQueen"
]
