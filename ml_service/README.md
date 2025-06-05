# ML Service - Bachelor's Thesis Demo

A comprehensive Machine Learning service demonstrating three key ML applications in e-commerce platforms. This service is designed for research and educational purposes, showcasing practical implementations of ML techniques in real-world scenarios.

## 🎯 Overview

This ML service implements three core functionalities:

### 1. 🛡️ HTTP Request Security Analysis
- **Purpose**: Detect malicious HTTP requests (XSS, SQLi, CSRF attacks)
- **Techniques**: TF-IDF + Naive Bayes, Logistic Regression, Isolation Forest, Pattern Matching
- **Use Case**: Real-time security monitoring and threat detection

### 2. 🎯 Personalized Recommendations & Dynamic Deals
- **Purpose**: Generate personalized product recommendations and dynamic pricing
- **Techniques**: Collaborative Filtering, Content-based Filtering, Clustering, Hybrid Methods
- **Use Case**: E-commerce personalization and targeted marketing

### 3. 🤖 Product Automation (Smart Catalog Enrichment)
- **Purpose**: Automatically tag and price new products
- **Techniques**: TF-IDF + Similarity, Multi-label Classification, Price Prediction
- **Use Case**: Automated product onboarding and catalog management

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Navigate to the ml_service directory**:
```bash
cd ml_service
```

2. **Activate virtual environment**:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies** (if not already installed):
```bash
pip install -r requirements.txt
```

4. **Start the service**:
```bash
python main.py
```

The service will be available at: `http://localhost:8001`

## 📚 API Documentation

Once the service is running, visit:
- **Interactive API Docs**: `http://localhost:8001/docs`
- **Alternative Docs**: `http://localhost:8001/redoc`
- **Complete Demo Guide**: `http://localhost:8001/demo/complete-workflow`

## 🔬 Research Scenarios

### Getting Started with Demo Data

1. **Train All Models** (takes 2-5 minutes):
```bash
curl -X POST "http://localhost:8001/models/train-all"
```

2. **Check Training Status**:
```bash
curl "http://localhost:8001/models/status"
```

### 1. Security Analysis Research

#### Test Different Attack Patterns
```bash
# Get demo attack scenarios
curl "http://localhost:8001/security/demo-attacks"

# Analyze XSS attack
curl -X POST "http://localhost:8001/security/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/api/search?q=<script>alert(\"XSS\")</script>",
    "method": "GET",
    "user_agent": "Mozilla/5.0",
    "ip_address": "192.168.1.100"
  }'

# Analyze SQL Injection
curl -X POST "http://localhost:8001/security/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/api/products?id=1\" UNION SELECT password FROM users--",
    "method": "GET",
    "user_agent": "Mozilla/5.0",
    "ip_address": "10.0.0.50"
  }'
```

#### Research Questions:
- How accurate is the multi-layered detection approach?
- Which ML technique performs best for each attack type?
- How do false positive rates compare across methods?

### 2. Recommendation System Research

#### Test Regional and Demographic Differences
```bash
# Get demo users from different segments
curl "http://localhost:8001/recommendations/demo-users"

# Get recommendations for EU user (young)
curl -X POST "http://localhost:8001/recommendations/get" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "num_recommendations": 10,
    "strategy": "hybrid"
  }'

# Compare with different strategies
curl -X POST "http://localhost:8001/recommendations/get" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "strategy": "collaborative"
  }'
```

#### Analyze User Segments
```bash
# Get user clustering analysis
curl "http://localhost:8001/recommendations/segments"

# Get personalized deals
curl "http://localhost:8001/recommendations/deals/user_123"
```

#### Research Questions:
- How do recommendations differ across regions (EU vs NA vs Asia)?
- What's the impact of age groups on recommendation quality?
- How effective is the hybrid approach vs individual methods?
- How do user clusters correlate with purchasing behavior?

### 3. Product Automation Research

#### Test Automated Product Analysis
```bash
# Get demo products
curl "http://localhost:8001/products/demo-products"

# Analyze a new product
curl -X POST "http://localhost:8001/products/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Wireless Bluetooth Headphones",
    "description": "High-quality wireless headphones with noise cancellation, 30-hour battery life, and premium sound quality."
  }'

# Get category insights
curl "http://localhost:8001/products/category/electronics"
```

#### Research Questions:
- How accurate is automatic category prediction?
- What's the effectiveness of ML vs rule-based tag suggestions?
- How close are price estimations to market prices?
- Which product features most influence pricing?

## 📊 Key Features for Research

### 1. Multiple ML Techniques Comparison
- **Security**: Pattern matching vs ML-based detection
- **Recommendations**: Collaborative vs Content-based vs Hybrid
- **Products**: Rule-based vs ML tag suggestions

### 2. Real-world Scenarios
- **Regional Preferences**: EU (electronics, books) vs Asia (fashion, beauty)
- **Age Demographics**: Young (gaming, fashion) vs Senior (health, books)
- **Attack Patterns**: XSS, SQLi, CSRF with varying complexity

### 3. Comprehensive Analytics
- Model performance metrics
- User segmentation analysis
- Category-specific insights
- Confidence scores and explanations

## 🔧 Advanced Usage

### Custom Data Integration

You can train models with your own data by modifying the training methods:

```python
# Example: Train with custom security data
security_analyzer.train(
    requests=your_request_logs,
    labels=your_attack_labels
)

# Example: Train with custom product data
product_automation.train(
    products_data=your_product_catalog
)
```

### Model Information and Debugging

```bash
# Get detailed model information
curl "http://localhost:8001/security/model-info"
curl "http://localhost:8001/recommendations/model-info"
curl "http://localhost:8001/products/model-info"
```

## 📈 Research Metrics

### Security Analysis Metrics
- **Accuracy**: Overall detection rate
- **Precision/Recall**: Per attack type
- **False Positive Rate**: Normal requests flagged as attacks
- **Response Time**: Analysis speed

### Recommendation Metrics
- **Diversity**: Variety in recommended categories
- **Coverage**: Percentage of catalog recommended
- **Novelty**: New vs familiar recommendations
- **Regional Relevance**: Alignment with regional preferences

### Product Automation Metrics
- **Category Accuracy**: Correct category predictions
- **Tag Relevance**: Quality of suggested tags
- **Price Accuracy**: Estimation vs actual prices
- **Processing Speed**: Analysis time per product

## 🎓 Educational Value

This implementation demonstrates:

1. **Classical ML Techniques**: Naive Bayes, Logistic Regression, k-NN
2. **Unsupervised Learning**: Clustering, Isolation Forest
3. **NLP Applications**: TF-IDF, text similarity
4. **Hybrid Systems**: Combining multiple approaches
5. **Real-world Challenges**: Data sparsity, cold start problems
6. **Evaluation Metrics**: Precision, recall, accuracy, confidence

## 🔍 Troubleshooting

### Common Issues

1. **Models not training**: Check if virtual environment is activated
2. **Memory errors**: Reduce sample data size in model parameters
3. **Import errors**: Ensure all dependencies are installed
4. **Port conflicts**: Change port in `main.py` if 8001 is occupied

### Performance Optimization

- Models use lightweight algorithms suitable for demonstration
- Sample data sizes are configurable
- Background training prevents API blocking
- Caching can be added for production use

## 📝 Research Documentation

For your bachelor's thesis, document:

1. **Methodology**: Which ML techniques were chosen and why
2. **Experiments**: Different scenarios tested and results
3. **Comparisons**: Performance across different approaches
4. **Insights**: What the data reveals about user behavior/security threats
5. **Limitations**: Current constraints and future improvements

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Experiment with different ML algorithms
- Add new evaluation metrics
- Extend the demo scenarios
- Improve the documentation

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

---

**Happy researching! 🎓✨**

For questions or issues, check the API documentation at `/docs` or review the code comments for detailed explanations of each ML technique. 