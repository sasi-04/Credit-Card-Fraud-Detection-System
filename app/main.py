from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Any
import json
from pydantic import BaseModel

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None

class TransactionResponse(BaseModel):
    amount: float
    is_international: int
    is_online: int
    fraud_prediction: int
    fraud_probability: float
    explanation: str

class AnalysisResponse(BaseModel):
    total_transactions: int
    fraudulent_transactions: int
    safe_transactions: int
    fraud_percentage: float
    high_risk_transactions: int
    avg_fraud_amount: float
    probability_distribution: Dict[str, int]
    transactions: List[TransactionResponse]

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'fraud_model.pkl')
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'scaler.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
        return True
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return False

def get_explanation(probability: float, features: Dict[str, Any]) -> str:
    """Generate explanation for fraud prediction"""
    if probability > 0.7:
        return f"This transaction was flagged as fraud because the ML model predicted a high fraud probability ({probability:.1%}) based on learned patterns from historical data. Key risk factors: amount=${features['amount']:.2f}, international={features['is_international']}, online={features['is_online']}."
    else:
        return f"This transaction was considered safe with a low fraud probability ({probability:.1%}) based on the ML model's analysis."

def analyze_transactions(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive analysis on transactions"""
    total_transactions = len(df)
    fraudulent_transactions = df['fraud_prediction'].sum()
    safe_transactions = total_transactions - fraudulent_transactions
    fraud_percentage = (fraudulent_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    
    # High-risk transactions (probability > 0.5 but < 0.7)
    high_risk_transactions = len(df[(df['fraud_probability'] > 0.5) & (df['fraud_probability'] <= 0.7)])
    
    # Average transaction amount for fraud
    fraud_df = df[df['fraud_prediction'] == 1]
    avg_fraud_amount = fraud_df['amount'].mean() if len(fraud_df) > 0 else 0
    
    # Probability distribution
    prob_bins = {
        "0-20%": len(df[df['fraud_probability'] <= 0.2]),
        "21-40%": len(df[(df['fraud_probability'] > 0.2) & (df['fraud_probability'] <= 0.4)]),
        "41-60%": len(df[(df['fraud_probability'] > 0.4) & (df['fraud_probability'] <= 0.6)]),
        "61-80%": len(df[(df['fraud_probability'] > 0.6) & (df['fraud_probability'] <= 0.8)]),
        "81-100%": len(df[df['fraud_probability'] > 0.8])
    }
    
    return {
        "total_transactions": total_transactions,
        "fraudulent_transactions": int(fraudulent_transactions),
        "safe_transactions": int(safe_transactions),
        "fraud_percentage": round(fraud_percentage, 2),
        "high_risk_transactions": high_risk_transactions,
        "avg_fraud_amount": round(avg_fraud_amount, 2),
        "probability_distribution": prob_bins
    }

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("Warning: Model could not be loaded. Please ensure train_model.py has been run.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Credit Card Fraud Detection API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/upload-csv", response_model=AnalysisResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file and perform fraud detection"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['amount', 'is_international', 'is_online']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Validate data types and ranges
        if not df['amount'].apply(lambda x: isinstance(x, (int, float)) and x > 0).all():
            raise HTTPException(status_code=400, detail="Amount must be positive numbers")
        
        if not df['is_international'].isin([0, 1]).all():
            raise HTTPException(status_code=400, detail="is_international must be 0 or 1")
        
        if not df['is_online'].isin([0, 1]).all():
            raise HTTPException(status_code=400, detail="is_online must be 0 or 1")
        
        # Prepare features for prediction
        X = df[required_columns]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        fraud_probabilities = model.predict_proba(X_scaled)[:, 1]
        fraud_predictions = (fraud_probabilities > 0.7).astype(int)
        
        # Add predictions to dataframe
        df['fraud_probability'] = fraud_probabilities
        df['fraud_prediction'] = fraud_predictions
        
        # Generate transaction responses with explanations
        transactions = []
        for idx, row in df.iterrows():
            features = {
                'amount': row['amount'],
                'is_international': row['is_international'],
                'is_online': row['is_online']
            }
            explanation = get_explanation(row['fraud_probability'], features)
            
            transaction = TransactionResponse(
                amount=float(row['amount']),
                is_international=int(row['is_international']),
                is_online=int(row['is_online']),
                fraud_prediction=int(row['fraud_prediction']),
                fraud_probability=float(row['fraud_probability']),
                explanation=explanation
            )
            transactions.append(transaction)
        
        # Perform analysis
        analysis = analyze_transactions(df)
        
        # Create response
        response = AnalysisResponse(
            **analysis,
            transactions=transactions
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-single")
async def predict_single(amount: float, is_international: int, is_online: int):
    """Predict fraud for a single transaction"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    # Validate inputs
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    if is_international not in [0, 1]:
        raise HTTPException(status_code=400, detail="is_international must be 0 or 1")
    if is_online not in [0, 1]:
        raise HTTPException(status_code=400, detail="is_online must be 0 or 1")
    
    try:
        # Prepare features
        X = np.array([[amount, is_international, is_online]])
        X_scaled = scaler.transform(X)
        
        # Make prediction
        fraud_probability = model.predict_proba(X_scaled)[0, 1]
        fraud_prediction = 1 if fraud_probability > 0.7 else 0
        
        features = {
            'amount': amount,
            'is_international': is_international,
            'is_online': is_online
        }
        explanation = get_explanation(fraud_probability, features)
        
        return {
            "amount": amount,
            "is_international": is_international,
            "is_online": is_online,
            "fraud_prediction": int(fraud_prediction),
            "fraud_probability": float(fraud_probability),
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
