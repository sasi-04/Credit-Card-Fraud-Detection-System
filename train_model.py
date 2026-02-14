import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_synthetic_data(n_samples=10000):
    """Generate synthetic credit card transaction data for training"""
    np.random.seed(42)
    
    # Generate realistic transaction amounts
    # Most transactions are small, few are large
    amounts = np.random.exponential(scale=1000, size=n_samples)
    amounts = np.clip(amounts, 10, 100000)
    
    # Generate binary features
    is_international = np.random.binomial(1, 0.2, n_samples)  # 20% international
    is_online = np.random.binomial(1, 0.7, n_samples)  # 70% online
    
    # Create fraud patterns
    fraud = np.zeros(n_samples)
    
    # Higher fraud probability for international transactions
    fraud_prob = np.zeros(n_samples)
    fraud_prob += is_international * 0.3
    fraud_prob += is_online * 0.1
    fraud_prob += (amounts > 50000) * 0.4
    fraud_prob += (amounts > 20000) * 0.2
    
    # Add some randomness
    fraud_prob += np.random.normal(0, 0.1, n_samples)
    fraud_prob = np.clip(fraud_prob, 0, 1)
    
    # Generate fraud labels based on probability
    fraud = np.random.binomial(1, fraud_prob)
    
    # Ensure we have some fraud cases (around 5-10%)
    if fraud.sum() < n_samples * 0.05:
        additional_fraud = np.random.choice(
            np.where(fraud == 0)[0], 
            size=int(n_samples * 0.05) - fraud.sum(), 
            replace=False
        )
        fraud[additional_fraud] = 1
    
    df = pd.DataFrame({
        'amount': amounts,
        'is_international': is_international,
        'is_online': is_online,
        'fraud': fraud
    })
    
    return df

def train_model():
    """Train and save the fraud detection model"""
    print("Generating synthetic training data...")
    df = generate_synthetic_data(10000)
    
    # Prepare features and target
    X = df[['amount', 'is_international', 'is_online']]
    y = df['fraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = rf_model.predict(X_test_scaled)
    y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model and scaler
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(rf_model, os.path.join(model_dir, 'fraud_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print(f"\nModel saved to {model_dir}/fraud_model.pkl")
    print(f"Scaler saved to {model_dir}/scaler.pkl")
    
    # Save sample data for testing
    df.head(100).to_csv('data/sample_transactions.csv', index=False)
    print("Sample data saved to data/sample_transactions.csv")
    
    return rf_model, scaler

def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('model/fraud_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("Model files not found. Please run train_model() first.")
        return None, None

if __name__ == "__main__":
    train_model()
