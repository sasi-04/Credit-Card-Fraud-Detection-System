#!/usr/bin/env python3
"""
Test script for Credit Card Fraud Detection System
Validates all major components and functionality
"""

import requests
import pandas as pd
import json
import time
import os

API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test API health endpoint"""
    print("Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: API is healthy - Model loaded: {data.get('model_loaded', False)}")
            return True
        else:
            print(f"FAIL: API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: API connection error: {str(e)}")
        return False

def test_single_prediction():
    """Test single transaction prediction"""
    print("\n🔍 Testing Single Transaction Prediction...")
    
    test_cases = [
        {"amount": 1000, "is_international": 0, "is_online": 0, "expected": "safe"},
        {"amount": 50000, "is_international": 1, "is_online": 1, "expected": "potentially risky"},
        {"amount": 100000, "is_international": 1, "is_online": 1, "expected": "high risk"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            params = {
                "amount": case["amount"],
                "is_international": case["is_international"],
                "is_online": case["is_online"]
            }
            
            response = requests.post(f"{API_BASE_URL}/predict-single", params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                fraud_pred = result["fraud_prediction"]
                fraud_prob = result["fraud_probability"]
                
                print(f"  Test {i}: ${case['amount']:,} (Int:{case['is_international']}, Online:{case['is_online']})")
                print(f"    Prediction: {'FRAUD' if fraud_pred == 1 else 'SAFE'} (Probability: {fraud_prob:.1%})")
                print(f"    Expected: {case['expected']}")
                print(f"    ✅ Success")
            else:
                print(f"  Test {i}: ❌ Failed - {response.status_code}")
                
        except Exception as e:
            print(f"  Test {i}: ❌ Error - {str(e)}")
    
    return True

def test_csv_upload():
    """Test CSV upload functionality"""
    print("\n🔍 Testing CSV Upload...")
    
    # Test with sample data
    csv_files = [
        "data/test_transactions.csv",
        "data/large_test_dataset.csv"
    ]
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"  ⚠️  File not found: {csv_file}")
            continue
            
        print(f"  Testing with {csv_file}...")
        
        try:
            with open(csv_file, 'rb') as f:
                files = {'file': (os.path.basename(csv_file), f, 'text/csv')}
                response = requests.post(f"{API_BASE_URL}/upload-csv", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"    Total transactions: {result['total_transactions']}")
                print(f"    Fraudulent: {result['fraudulent_transactions']}")
                print(f"    Safe: {result['safe_transactions']}")
                print(f"    Fraud percentage: {result['fraud_percentage']:.2f}%")
                print(f"    High risk: {result['high_risk_transactions']}")
                print(f"    ✅ CSV processing successful")
                
                # Validate response structure
                required_fields = ['total_transactions', 'fraudulent_transactions', 
                                 'safe_transactions', 'fraud_percentage', 'transactions']
                missing_fields = [field for field in required_fields if field not in result]
                
                if missing_fields:
                    print(f"    ⚠️  Missing response fields: {missing_fields}")
                else:
                    print(f"    ✅ Response structure valid")
                    
            else:
                print(f"    ❌ CSV upload failed: {response.status_code}")
                if response.headers.get('content-type', '').startswith('application/json'):
                    error_data = response.json()
                    print(f"    Error: {error_data.get('detail', 'Unknown error')}")
                    
        except Exception as e:
            print(f"    ❌ Error processing {csv_file}: {str(e)}")
    
    return True

def test_model_files():
    """Test if model files exist and are valid"""
    print("\n🔍 Testing Model Files...")
    
    model_files = [
        "model/fraud_model.pkl",
        "model/scaler.pkl"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            print(f"  ✅ {model_file} exists ({file_size:,} bytes)")
        else:
            print(f"  ❌ {model_file} missing")
            return False
    
    return True

def test_data_files():
    """Test if sample data files exist"""
    print("\n🔍 Testing Data Files...")
    
    data_files = [
        "data/test_transactions.csv",
        "data/large_test_dataset.csv"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            try:
                df = pd.read_csv(data_file)
                print(f"  ✅ {data_file} exists ({len(df)} rows)")
                
                # Validate columns
                required_cols = ['amount', 'is_international', 'is_online']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"    ⚠️  Missing columns: {missing_cols}")
                else:
                    print(f"    ✅ Valid column structure")
                    
            except Exception as e:
                print(f"  ❌ Error reading {data_file}: {str(e)}")
        else:
            print(f"  ❌ {data_file} missing")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Credit Card Fraud Detection System Tests")
    print("=" * 60)
    
    # Test components
    tests = [
        ("Model Files", test_model_files),
        ("Data Files", test_data_files),
        ("API Health", test_api_health),
        ("Single Prediction", test_single_prediction),
        ("CSV Upload", test_csv_upload),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        print("\n📋 Next Steps:")
        print("1. FastAPI Backend: http://localhost:8000")
        print("2. Streamlit Frontend: http://localhost:8501")
        print("3. API Documentation: http://localhost:8000/docs")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
