#  AI Luxury Brand Analysis Platform

A comprehensive AI-powered platform for analyzing luxury fashion products, providing market insights, trend analysis, and strategic recommendations for luxury brands.

## 🚀 Features

- **Product Analysis**: Comprehensive analysis of luxury fashion products using AI models
- **Market Positioning**: Intelligent market positioning analysis and competitive landscape assessment
- **Trend Analysis**: Real-time trend alignment scoring and style longevity predictions
- **Price Analysis**: Value scoring and pricing recommendations based on market data
- **Customer Insights**: Target demographic analysis and customer persona generation
- **Seasonal Analysis**: Demand pattern analysis and peak season identification
- **REST API**: FastAPI-based backend for seamless integration
- **Web Interface**: Beautiful Streamlit-based user interface
- **Advanced Web Scraping**: Real-time data collection from luxury fashion websites using multiple methods
- **Google Custom Search**: Integration with Google Custom Search API for comprehensive product research
- **Fuzzy Matching**: Intelligent product matching to find closest results when exact matches aren't available

## 📁 Project Structure

```
AI-Luxury-Brand/
├── src/                          # Source code
│   ├── api/                      # FastAPI backend
│   │   └── main.py              # Main API endpoints
│   ├── models/                   # AI models
│   │   └── gemini_fashion.py    # Gemini AI integration
│   ├── analysis/                 # Analysis engines
│   │   └── product_analyzer.py  # Main product analysis engine
│   ├── agents/                   # AI agents
│   │   └── market_agent.py      # Market research agent
│   ├── utils/                    # Utility modules
│   │   ├── database.py          # Database utilities
│   │   ├── sentiment_analyzer.py # Sentiment analysis
│   │   └── text_processing.py   # Text processing utilities
│   └── ui/                       # User interface
│       └── streamlit_app.py     # Streamlit web app
├── datasets/                     # Data files
│   ├── amazon_co-ecommerce_sample.csv
│   ├── data_amazon.xlsx_-_Sheet1.csv
│   └── handm.csv
├── configs/                      # Configuration files
│   └── config.py                # Application configuration
├── myenv/                        # Python virtual environment
├── .env                          # Environment variables
├── .env.example                  # Environment variables template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## �️ Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI-Luxury-Brand
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/Mac
python -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

Edit `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=sqlite:///./fashion_analysis.db
DEBUG=True
```

## 🚀 Running the Application

### Option 1: Using Scripts

#### Start the FastAPI Backend
```bash
# Windows
C:/path/to/your/project/myenv/Scripts/python.exe -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Linux/Mac
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Start the Streamlit Frontend
```bash
# Windows  
C:/path/to/your/project/myenv/Scripts/python.exe -m streamlit run src/ui/streamlit_app.py

# Linux/Mac
streamlit run src/ui/streamlit_app.py
```

### Option 2: Using VS Code Tasks
If using VS Code, you can use the pre-configured task:
- Open Command Palette (`Ctrl+Shift+P`)
- Type "Tasks: Run Task"
- Select "Start Fashion API Server"

## 📱 Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8501`
2. Use the Streamlit interface to analyze products
3. Input product details (name, brand, price, description)
4. View comprehensive analysis results

### API Endpoints
The FastAPI server runs on `http://localhost:8000`

#### Key Endpoints:
- `GET /health` - Health check
- `POST /analyze/product` - Analyze a product
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

#### Example API Usage:
```python
import requests

# Analyze a product
response = requests.post("http://localhost:8000/analyze/product", json={
    "name": "Cashmere Sweater",
    "brand": "Luxury Brand",
    "price": 850.0,
    "description": "Premium cashmere sweater with elegant design..."
})

analysis = response.json()
print(f"Overall Score: {analysis['overall_score']}")
```

## 🔧 Configuration

### API Keys
- **Gemini API**: Required for AI-powered analysis
- Get your API key from [Google AI Studio](https://makersuite.google.com/)

### Database
- Default: SQLite (for development)
- Production: Configure PostgreSQL or other databases in `configs/config.py`

## 📊 Datasets

The project includes sample datasets for testing and analysis:
- `amazon_co-ecommerce_sample.csv` - Amazon fashion product samples
- `data_amazon.xlsx_-_Sheet1.csv` - Amazon product data export
- `handm.csv` - H&M fashion product data

## 🧪 Analysis Features

### Product Analysis Engine
- **Trend Alignment**: Scores products based on current fashion trends
- **Market Positioning**: Analyzes competitive landscape and positioning
- **Value Assessment**: Evaluates price-to-value ratio
- **Feature Analysis**: Identifies key features and gaps
- **Target Demographics**: Determines ideal customer segments

### AI Integration
- **Gemini AI**: Advanced natural language processing for product descriptions
- **Sentiment Analysis**: Customer sentiment evaluation
- **Text Processing**: Advanced text analysis and feature extraction

## � API Documentation

Once the FastAPI server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🛠️ Development

### Adding New Features
1. Create new modules in the appropriate `src/` subdirectory
2. Update `requirements.txt` if new dependencies are added
3. Update this README with new features

### Code Structure
- **API Layer**: `src/api/` - FastAPI endpoints and request handling
- **Analysis Layer**: `src/analysis/` - Core analysis algorithms
- **Models Layer**: `src/models/` - AI model integrations
- **Utils Layer**: `src/utils/` - Shared utilities and helpers
- **UI Layer**: `src/ui/` - User interface components

## � Performance

- **Response Time**: Typical analysis completes in 2-5 seconds
- **Concurrent Users**: Supports multiple simultaneous analyses
- **Scalability**: Designed for horizontal scaling with load balancers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

#### "Cannot connect to the API server"
- Ensure the FastAPI server is running on port 8000
- Check that no firewall is blocking the connection
- Verify the API server started without errors

#### "Module not found" errors
- Activate your virtual environment: `myenv\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt`

#### Gemini API errors
- Verify your API key in the `.env` file
- Check your API quota and billing status
- Ensure the API key has the necessary permissions

### Getting Help
- Check the [Issues](../../issues) page for known problems
- Create a new issue if you encounter a bug
- Review the API documentation at `/docs` endpoint

## 🔮 Future Enhancements

- [ ] Integration with more AI models (GPT-4, Claude)
- [ ] Real-time market data integration
- [ ] Advanced visualization dashboards
- [ ] Mobile app development
- [ ] Multi-language support
- [ ] Enhanced recommendation algorithms
- [ ] Social media trend integration

---

**Made with ❤️ for the luxury fashion industry**
