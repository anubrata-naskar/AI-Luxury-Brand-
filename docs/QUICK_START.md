# AI Luxury Fashion Analysis System

Welcome to the AI Luxury Fashion Brand Analysis System! This project provides comprehensive analysis of luxury fashion products using AI models and market intelligence.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Luxury-Brand-

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and model paths
# MODEL_PATH=./models/llama-3b-fashion
# OPENAI_API_KEY=your_key_here
# SERP_API_KEY=your_key_here
```

### 3. Run the API Server

```bash
uvicorn src.api.main:app --reload
```

### 4. Launch Web Interface

```bash
streamlit run src/ui/streamlit_app.py
```

### 5. Explore the Notebook

Open `notebooks/fashion_analysis_demo.ipynb` in Jupyter Lab or VS Code.

## Features

- 🔍 **Product Analysis**: Deep insights from product descriptions
- 📊 **Market Research**: Automated competitor discovery and analysis  
- 😊 **Sentiment Analysis**: Customer sentiment from reviews
- 📈 **Trend Analysis**: Current and emerging fashion trends
- 💰 **Price Optimization**: Competitive pricing recommendations
- 🎯 **Target Audience**: Demographic and persona analysis

## Usage Examples

### Python API

```python
from src.analysis.product_analyzer import ProductAnalyzer

analyzer = ProductAnalyzer()
result = analyzer.analyze_product({
    "name": "Cashmere Sweater",
    "brand": "Luxury Brand",
    "price": 450,
    "description": "Premium cashmere..."
})

print(f"Overall Score: {result.overall_score}")
```

### REST API

```bash
curl -X POST "http://localhost:8000/analyze/product" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Cashmere Sweater",
       "brand": "Luxury Brand", 
       "price": 450,
       "description": "Premium cashmere..."
     }'
```

### Web Interface

Visit `http://localhost:8501` after running the Streamlit app.

## Project Structure

```
├── src/
│   ├── models/          # Fine-tuned model integration
│   ├── analysis/        # Product analysis engine
│   ├── agents/          # AI research agents
│   ├── api/            # FastAPI endpoints
│   └── ui/             # Streamlit interface
├── notebooks/          # Jupyter analysis demos
├── tests/              # Test suite
├── configs/            # Configuration files
└── data/               # Data storage
```

## Model Training (Colab)

Since local hardware constraints prevent fine-tuning, use Google Colab:

1. **Upload datasets** to Colab
2. **Fine-tune LLaMA-3B** using provided training scripts
3. **Download model weights** to local `models/` directory
4. **Update model paths** in configuration

### Training Datasets Used

- Fashion Product Feedback Dataset (OpenDataBay, 2025)
- Amazon Fashion Reviews subset
- Data4Fashion Text Dataset
- Myntra Product Data (Kaggle)
- H&M Product Descriptions (Kaggle)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fashion_analysis.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues:
- 📧 Email: support@fashionai.com
- 💬 Discord: [Fashion AI Community]
- 📖 Documentation: [docs.fashionai.com]

## Roadmap

- [ ] Multi-modal analysis (text + images)
- [ ] Real-time trend detection
- [ ] Advanced competitor tracking
- [ ] Mobile app development
- [ ] Enterprise dashboard

---

Made with ❤️ for the fashion industry
