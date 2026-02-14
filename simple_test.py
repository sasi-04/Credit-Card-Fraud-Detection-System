#!/usr/bin/env python3
"""
Simple test script for Credit Card Fraud Detection System
"""

import requests
import pandas as pd
import os

API_BASE_URL = "http://localhost:8000"

def test_system():
    print("Starting Credit Card Fraud Detection System Tests")
    print("=" * 60)
    
    # Test 1: API Health
    print("Test 1: API Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: API is healthy - Model loaded: {data.get('model_loaded', False)}")
        else:
            print(f"FAIL: API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: API connection error: {str(e)}")
        return False
    
    # Test 2: Single Prediction
    print("\nTest 2: Single Transaction Prediction")
    try:
        params = {"amount": 50000, "is_international": 1, "is_online": 1}
        response = requests.post(f"{API_BASE_URL}/predict-single", params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"PASS: Prediction successful")
            print(f"  Amount: ${result['amount']:,.2f}")
            print(f"  Fraud Prediction: {'FRAUD' if result['fraud_prediction'] == 1 else 'SAFE'}")
            print(f"  Probability: {result['fraud_probability']:.1%}")
        else:
            print(f"FAIL: Single prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Single prediction error: {str(e)}")
        return False
    
    # Test 3: CSV Upload
    print("\nTest 3: CSV Upload")
    try:
        csv_file = "data/test_transactions.csv"
        if not os.path.exists(csv_file):
            print(f"FAIL: Test file not found: {csv_file}")
            return False
            
        with open(csv_file, 'rb') as f:
            files = {'file': (os.path.basename(csv_file), f, 'text/csv')}
            response = requests.post(f"{API_BASE_URL}/upload-csv", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"PASS: CSV processing successful")
            print(f"  Total transactions: {result['total_transactions']}")
            print(f"  Fraudulent: {result['fraudulent_transactions']}")
            print(f"  Safe: {result['safe_transactions']}")
            print(f"  Fraud percentage: {result['fraud_percentage']:.2f}%")
        else:
            print(f"FAIL: CSV upload failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: CSV upload error: {str(e)}")
        return False
    
    # Test 4: Model Files
    print("\nTest 4: Model Files")
    model_files = ["model/fraud_model.pkl", "model/scaler.pkl"]
    for model_file in model_files:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            print(f"PASS: {model_file} exists ({file_size:,} bytes)")
        else:
            print(f"FAIL: {model_file} missing")
            return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! System is ready for use.")
    print("\nNext Steps:")
    print("1. FastAPI Backend: http://localhost:8000")
    print("2. Streamlit Frontend: http://localhost:8501")
    print("3. API Documentation: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    success = test_system()
    exit(0 if success else 1)
