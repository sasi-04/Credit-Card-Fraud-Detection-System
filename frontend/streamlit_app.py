import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json

# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# API endpoint
API_BASE_URL = "http://localhost:8000"

def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Upload & Analyze", "Single Prediction", "About"])
    
    if page == "Upload & Analyze":
        upload_and_analyze_page()
    elif page == "Single Prediction":
        single_prediction_page()
    else:
        about_page()

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_and_analyze_page():
    st.header("📊 Upload & Analyze Transactions")
    
    # Check API health
    if not check_api_health():
        st.error("❌ API is not running. Please start the FastAPI server first.")
        st.code("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    st.success("✅ API is running")
    
    # File upload section
    st.subheader("📁 Upload CSV File")
    st.markdown("""
    **Expected CSV Format:**
    ```csv
    amount,is_international,is_online
    1200,0,0
    45000,1,1
    800,0,1
    ```
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Show file preview
        st.subheader("📋 File Preview")
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head(10))
        
        # Analyze button
        if st.button("🔍 Analyze Transactions", type="primary"):
            with st.spinner("Analyzing transactions..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Send to API
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(f"{API_BASE_URL}/upload-csv", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        display_analysis_results(result)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")

def display_analysis_results(result):
    """Display comprehensive analysis results"""
    st.success("✅ Analysis Complete!")
    
    # Summary metrics
    st.subheader("📈 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", result['total_transactions'])
    
    with col2:
        st.metric("Fraudulent", result['fraudulent_transactions'])
    
    with col3:
        st.metric("Safe", result['safe_transactions'])
    
    with col4:
        st.metric("Fraud %", f"{result['fraud_percentage']:.2f}%")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk", result['high_risk_transactions'])
    
    with col2:
        st.metric("Avg Fraud Amount", f"${result['avg_fraud_amount']:,.2f}")
    
    with col3:
        fraud_rate = (result['fraudulent_transactions'] / result['total_transactions']) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    # Create columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart: Fraud vs Safe
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Safe', 'Fraudulent'],
            values=[result['safe_transactions'], result['fraudulent_transactions']],
            hole=0.4,
            marker_colors=['#2E8B57', '#FF6B6B']
        )])
        fig_pie.update_layout(title="Fraud vs Safe Transactions")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Probability distribution
        prob_dist = result['probability_distribution']
        fig_bar = go.Figure(data=[go.Bar(
            x=list(prob_dist.keys()),
            y=list(prob_dist.values()),
            marker_color='#4A90E2'
        )])
        fig_bar.update_layout(
            title="Fraud Probability Distribution",
            xaxis_title="Probability Range",
            yaxis_title="Number of Transactions"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Transaction details
    st.subheader("📋 Transaction Details")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_fraud_only = st.checkbox("Show Fraudulent Transactions Only")
    with col2:
        show_high_risk = st.checkbox("Show High Risk Transactions (>50% probability)")
    
    # Filter transactions
    transactions = result['transactions']
    if show_fraud_only:
        transactions = [t for t in transactions if t['fraud_prediction'] == 1]
    elif show_high_risk:
        transactions = [t for t in transactions if t['fraud_probability'] > 0.5]
    
    if transactions:
        # Create DataFrame for display
        df_display = pd.DataFrame(transactions)
        
        # Format for display
        df_display['Amount'] = df_display['amount'].apply(lambda x: f"${x:,.2f}")
        df_display['International'] = df_display['is_international'].apply(lambda x: "Yes" if x == 1 else "No")
        df_display['Online'] = df_display['is_online'].apply(lambda x: "Yes" if x == 1 else "No")
        df_display['Fraud Probability'] = df_display['fraud_probability'].apply(lambda x: f"{x:.2%}")
        df_display['Prediction'] = df_display['fraud_prediction'].apply(lambda x: "🚨 FRAUD" if x == 1 else "✅ Safe")
        
        # Reorder columns
        display_columns = ['Amount', 'International', 'Online', 'Fraud Probability', 'Prediction', 'explanation']
        df_display = df_display[display_columns]
        
        # Display with color coding
        def highlight_fraud(row):
            if row['Prediction'] == '🚨 FRAUD':
                return ['background-color: #FF6B6B'] * len(row)
            else:
                return ['background-color: #2E8B57'] * len(row)
        
        styled_df = df_display.style.apply(highlight_fraud, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Download results
        st.subheader("💾 Download Results")
        csv_data = pd.DataFrame(transactions).to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="fraud_analysis_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No transactions match the selected filters.")

def single_prediction_page():
    st.header("🔍 Single Transaction Prediction")
    
    # Check API health
    if not check_api_health():
        st.error("❌ API is not running. Please start the FastAPI server first.")
        st.code("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    st.success("✅ API is running")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                value=1000.0,
                step=100.0,
                format="%.2f"
            )
            
            is_international = st.selectbox(
                "International Transaction?",
                options=[("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0],
                index=0
            )[1]
        
        with col2:
            is_online = st.selectbox(
                "Online Transaction?",
                options=[("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0],
                index=0
            )[1]
        
        submitted = st.form_submit_button("🔍 Predict Fraud", type="primary")
        
        if submitted:
            with st.spinner("Analyzing transaction..."):
                try:
                    # Send to API
                    params = {
                        "amount": amount,
                        "is_international": is_international,
                        "is_online": is_online
                    }
                    
                    response = requests.post(f"{API_BASE_URL}/predict-single", params=params)
                    
                    if response.status_code == 200:
                        result = response.json()
                        display_single_prediction(result)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")

def display_single_prediction(result):
    """Display single prediction result"""
    st.success("✅ Prediction Complete!")
    
    # Result card
    if result['fraud_prediction'] == 1:
        st.error("🚨 **FRAUD DETECTED**")
        risk_level = "High"
        risk_color = "#FF6B6B"
    else:
        st.success("✅ **SAFE TRANSACTION**")
        risk_level = "Low"
        risk_color = "#2E8B57"
    
    # Transaction details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Amount", f"${result['amount']:,.2f}")
    
    with col2:
        st.metric("International", "Yes" if result['is_international'] == 1 else "No")
    
    with col3:
        st.metric("Online", "Yes" if result['is_online'] == 1 else "No")
    
    # Probability gauge
    st.subheader("📊 Fraud Probability")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = result['fraud_probability'] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Probability (%)"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.subheader("📝 Explanation")
    st.info(result['explanation'])

def about_page():
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This Credit Card Fraud Detection system uses Machine Learning to identify potentially fraudulent transactions
    in real-time. The system analyzes transaction patterns and provides explainable predictions.
    
    ## 🤖 How It Works
    
    1. **Machine Learning Model**: Random Forest classifier trained on historical transaction data
    2. **Feature Analysis**: Analyzes transaction amount, international status, and online status
    3. **Probability Scoring**: Provides fraud probability scores with explanations
    4. **Real-time Processing**: Instant predictions for single transactions or bulk CSV uploads
    
    ## 📊 Key Features
    
    - **Bulk Analysis**: Upload CSV files with multiple transactions
    - **Single Prediction**: Analyze individual transactions
    - **Visual Analytics**: Charts and graphs for fraud patterns
    - **Explainable AI**: Clear explanations for each prediction
    - **Risk Assessment**: Probability-based fraud detection
    
    ## 🔧 Technical Stack
    
    - **Backend**: FastAPI with Python
    - **Machine Learning**: Scikit-learn (Random Forest)
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy
    
    ## 📈 Model Performance
    
    The model is trained to:
    - Achieve high accuracy in fraud detection
    - Minimize false positives
    - Provide probability-based predictions
    - Explain decision-making process
    
    ## 🚀 Getting Started
    
    1. **Train the Model**: Run `python train_model.py`
    2. **Start API Server**: Run `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
    3. **Launch Frontend**: Run `streamlit run frontend/streamlit_app.py`
    
    ## 📝 CSV Format
    
    Upload CSV files with the following format:
    ```csv
    amount,is_international,is_online
    1200,0,0
    45000,1,1
    800,0,1
    ```
    
    - **amount**: Transaction amount (positive number)
    - **is_international**: 0 for domestic, 1 for international
    - **is_online**: 0 for offline, 1 for online
    
    ## ⚠️ Important Notes
    
    - This is a demonstration system for educational purposes
    - The model uses synthetic training data
    - In production, use real historical transaction data
    - Always validate predictions with human review
    - Consider additional security measures for real applications
    """)

if __name__ == "__main__":
    main()
