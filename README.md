# 💳 Credit Card Fraud Detection System

A production-style Machine Learning application that detects fraudulent credit card transactions using supervised learning algorithms. The system provides real-time fraud detection with explainable AI predictions and comprehensive analytics.

## 🎯 Project Overview

This system demonstrates a complete end-to-end ML pipeline with:
- **Machine Learning Model**: Random Forest classifier for fraud detection
- **REST API**: FastAPI backend for transaction processing
- **Interactive Dashboard**: Streamlit frontend with visualizations
- **Explainable AI**: Clear explanations for each prediction
- **Bulk Processing**: CSV upload for analyzing multiple transactions

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   ML Model      │
│   Frontend      │◄──►│   Backend       │◄──►│   (Random       │
│                 │    │                 │    │   Forest)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌─────────┐            ┌─────────────┐         ┌─────────────┐
    │ Charts  │            │ CSV Upload  │         │ Joblib      │
    │ Tables  │            │ Validation  │         │ Model Files │
    │ Analytics│           │ Predictions │         │ Scaler     │
    └─────────┘            └─────────────┘         └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the ML model**
   ```bash
   python train_model.py
   ```
   This will:
   - Generate synthetic training data
   - Train a Random Forest classifier
   - Save the model and scaler to `model/` directory
   - Create sample data in `data/` directory

4. **Start the FastAPI backend**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   The API will be available at `http://localhost:8000`

5. **Launch the Streamlit frontend**
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
   The dashboard will open in your browser at `http://localhost:8501`

## 📊 Features

### 🔍 Fraud Detection
- **Binary Classification**: Predicts fraud vs safe transactions
- **Probability Scoring**: Provides fraud probability (0-100%)
- **Threshold-based**: Uses 70% probability threshold for fraud detection
- **Explainable AI**: Clear explanations for each prediction

### 📈 Analytics & Visualization
- **Summary Metrics**: Total transactions, fraud count, fraud percentage
- **Interactive Charts**: Pie charts, bar graphs, probability distributions
- **Risk Assessment**: High-risk transaction identification
- **Transaction Details**: Detailed breakdown with explanations

### 🔄 Bulk Processing
- **CSV Upload**: Process multiple transactions at once
- **Data Validation**: Validates input format and data types
- **Batch Predictions**: Efficient processing of large datasets
- **Export Results**: Download analysis results as CSV

### 🎯 Single Prediction
- **Real-time Analysis**: Instant fraud prediction for individual transactions
- **Interactive Form**: User-friendly input interface
- **Visual Feedback**: Gauge charts for probability visualization
- **Detailed Explanations**: Risk factors and decision rationale

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app/
│   └── main.py                 # FastAPI backend application
├── model/
│   ├── fraud_model.pkl         # Trained ML model
│   └── scaler.pkl              # Feature scaler
├── frontend/
│   └── streamlit_app.py        # Streamlit dashboard
├── data/
│   ├── sample_transactions.csv # Sample training data
│   ├── test_transactions.csv   # Test data (30 transactions)
│   └── large_test_dataset.csv  # Large test dataset (150 transactions)
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Technical Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Scikit-learn**: Machine learning library for Random Forest
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Joblib**: Model serialization and parallel processing

### Frontend
- **Streamlit**: Interactive web applications for data science
- **Plotly**: Interactive visualization library
- **Requests**: HTTP library for API calls

### Machine Learning
- **Random Forest**: Ensemble learning method for classification
- **Feature Scaling**: StandardScaler for data preprocessing
- **Probability Threshold**: 70% for fraud detection

## 📋 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```
Returns API status and model loading status.

#### Single Transaction Prediction
```http
POST /predict-single?amount=1000&is_international=1&is_online=1
```
Predicts fraud for a single transaction.

#### CSV Upload and Analysis
```http
POST /upload-csv
Content-Type: multipart/form-data
```
Uploads CSV file for bulk fraud analysis.

### Response Format

```json
{
  "total_transactions": 100,
  "fraudulent_transactions": 15,
  "safe_transactions": 85,
  "fraud_percentage": 15.0,
  "high_risk_transactions": 8,
  "avg_fraud_amount": 45000.50,
  "probability_distribution": {
    "0-20%": 70,
    "21-40%": 10,
    "41-60%": 5,
    "61-80%": 8,
    "81-100%": 7
  },
  "transactions": [
    {
      "amount": 1200.0,
      "is_international": 0,
      "is_online": 1,
      "fraud_prediction": 0,
      "fraud_probability": 0.15,
      "explanation": "This transaction was considered safe..."
    }
  ]
}
```

## 📄 CSV Format

### Input Format
Upload CSV files with the following structure:

```csv
amount,is_international,is_online
1200,0,0
45000,1,1
800,0,1
90000,1,1
```

### Field Descriptions
- **amount**: Transaction amount (positive float)
- **is_international**: 0 for domestic, 1 for international
- **is_online**: 0 for offline/in-person, 1 for online transaction

### Sample Data Files
- `data/test_transactions.csv`: 30 sample transactions
- `data/large_test_dataset.csv`: 150 transactions for testing

## 🤖 Model Details

### Training Data
- **Synthetic Generation**: 10,000 synthetic transactions
- **Feature Distribution**: Realistic transaction patterns
- **Fraud Patterns**: International, high-value, online transactions
- **Class Balance**: ~5-10% fraud rate (realistic)

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Features**: Amount, International Status, Online Status
- **Accuracy**: ~85% on test data
- **Threshold**: 70% probability for fraud detection

### Feature Importance
1. **Transaction Amount**: Higher amounts indicate higher risk
2. **International Status**: International transactions are riskier
3. **Online Status**: Online transactions have moderate risk

## 🎨 Dashboard Features

### Pages
1. **Upload & Analyze**: Bulk CSV processing with visualizations
2. **Single Prediction**: Individual transaction analysis
3. **About**: System information and documentation

### Visualizations
- **Pie Chart**: Fraud vs Safe transaction distribution
- **Bar Chart**: Fraud probability distribution
- **Gauge Chart**: Single transaction probability
- **Data Tables**: Detailed transaction information

### Interactive Features
- **File Upload**: Drag-and-drop CSV upload
- **Real-time Predictions**: Instant analysis
- **Filter Options**: Show fraud/high-risk transactions only
- **Export Results**: Download analysis as CSV

## 🔒 Security Considerations

### Current Implementation
- **Input Validation**: Validates data types and ranges
- **Error Handling**: Comprehensive error messages
- **CORS Configuration**: Configured for development

### Production Recommendations
- **Authentication**: Add API key authentication
- **Rate Limiting**: Implement request rate limits
- **HTTPS**: Use SSL/TLS encryption
- **Database**: Store transaction history securely
- **Logging**: Implement audit logging
- **Monitoring**: Add health checks and alerts

## 🚀 Deployment

### Development
```bash
# Backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
streamlit run frontend/streamlit_app.py
```

### Production (Docker)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
- `API_BASE_URL`: Backend API URL (default: http://localhost:8000)
- `MODEL_PATH`: Path to trained model files
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

## 🧪 Testing

### Unit Tests
```bash
# Test model training
python -m pytest tests/test_model.py

# Test API endpoints
python -m pytest tests/test_api.py

# Test frontend components
python -m pytest tests/test_frontend.py
```

### Integration Testing
1. **Model Training**: Verify model accuracy > 80%
2. **API Health**: Check `/health` endpoint
3. **CSV Upload**: Test with sample data files
4. **Single Prediction**: Test edge cases

### Test Data
- Use provided sample CSV files
- Test with various transaction amounts
- Verify international/online combinations
- Check error handling for invalid data

## 📈 Performance Metrics

### Model Performance
- **Accuracy**: ~85%
- **Precision**: Varies by fraud threshold
- **Recall**: Optimized for fraud detection
- **F1-Score**: Balanced precision/recall

### System Performance
- **API Response Time**: < 1 second for 1000 transactions
- **Memory Usage**: < 500MB for model and processing
- **Scalability**: Handles large CSV files efficiently

## 🔧 Configuration

### Model Parameters
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
```

### Fraud Threshold
- **Default**: 70% probability
- **Adjustable**: Modify in `app/main.py`
- **Trade-off**: Higher threshold = fewer false positives

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # Solution: Train the model first
   python train_model.py
   ```

2. **API Connection Error**
   ```bash
   # Solution: Check if API is running
   curl http://localhost:8000/health
   ```

3. **CSV Upload Fails**
   - Check CSV format (amount, is_international, is_online)
   - Ensure all values are numeric
   - Verify no headers in data rows

4. **Memory Issues**
   - Reduce batch size for large files
   - Use streaming for very large datasets
   - Monitor system resources

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload --log-level debug
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Include error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- FastAPI for modern API framework
- Streamlit for interactive data applications
- Plotly for beautiful visualizations

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review API documentation
3. Create an issue in the repository
4. Contact the development team

---

**Note**: This is a demonstration system for educational purposes. For production use, ensure proper security measures, real training data, and comprehensive testing.
