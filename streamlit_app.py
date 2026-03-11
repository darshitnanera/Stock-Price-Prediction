
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import (
    load_data, explore_data, handle_missing_values, handle_outliers, 
    add_features, preprocess_data, run_prediction, 
    plot_actual_vs_predicted, plot_scatter
)

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Stock Price Prediction Pipeline")
st.markdown("---")

# Sidebar for configuration
st.sidebar.header("Configuration")

# File upload or sample data
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV", "Use Sample Data (TSLA)"])

df = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    # Use sample TSLA data - try multiple paths for local and cloud deployment
    try:
        # Try different paths for local and cloud deployment
        import os
        possible_paths = [
            '../Datasets/TSLA.csv',
            'Datasets/TSLA.csv',
            './Datasets/TSLA.csv',
            '/app/Datasets/TSLA.csv',
            'TSLA.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = load_data(path)
                st.sidebar.success(f"Loaded TSLA sample data from {path}!")
                break
            except:
                continue
        
        if df is None:
            # Last resort: try to create sample data or show error
            st.sidebar.error("Could not load sample data. Please upload a CSV file.")
            st.info("💡 Tip: Download TSLA.csv from Yahoo Finance or upload your own data.")
    except Exception as e:
        st.sidebar.error(f"Could not load sample data: {str(e)}")

if df is not None:
    # ==================== STEP 1: Data Exploration ====================
    st.header("1️⃣ Data Exploration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        st.dataframe(df.dtypes, use_container_width=True)
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values visualization
    st.subheader("Missing Values Analysis")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.write(missing[missing > 0])
        fig_missing, ax = plt.subplots()
        missing[missing > 0].plot(kind='bar', ax=ax, color='red')
        ax.set_title('Missing Values per Column')
        ax.set_ylabel('Count')
        st.pyplot(fig_missing)
    else:
        st.success("No missing values found!")
    
    st.markdown("---")
    
    # ==================== STEP 2: Missing Values Handling ====================
    st.header("2️⃣ Missing Values Handling")
    
    col1, col2 = st.columns(2)
    with col1:
        missing_strategy = st.selectbox("Select Missing Value Strategy", 
            ['forward_fill', 'mean', 'median', 'drop'], index=0)
    with col2:
        st.info(f"Strategy: {missing_strategy}")
    
    df_cleaned = handle_missing_values(df.copy(), strategy=missing_strategy)
    st.success(f"Missing values handled! Shape: {df_cleaned.shape}")
    
    st.markdown("---")
    
    # ==================== STEP 3: Outlier Detection ====================
    st.header("3️⃣ Outlier Detection & Removal")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        outlier_column = st.selectbox("Select Column for Outlier Detection", 
            df_cleaned.select_dtypes(include=[np.number]).columns.tolist())
    with col2:
        outlier_method = st.selectbox("Method", ['iqr', 'zscore'], index=0)
    with col3:
        outlier_threshold = st.slider("Threshold", 1.0, 3.0, 1.5, 0.1)
    
    # Show outlier visualization before removal
    fig_outlier, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Box plot
    axes[0].boxplot(df_cleaned[outlier_column].dropna())
    axes[0].set_title(f'Box Plot - {outlier_column}')
    axes[0].set_ylabel(outlier_column)
    
    # Histogram
    axes[1].hist(df_cleaned[outlier_column].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Distribution - {outlier_column}')
    axes[1].set_xlabel(outlier_column)
    axes[1].set_ylabel('Frequency')
    
    st.pyplot(fig_outlier)
    
    # Remove outliers
    df_no_outliers = handle_outliers(df_cleaned.copy(), outlier_column, method=outlier_method, threshold=outlier_threshold)
    st.success(f"Outliers removed! Rows before: {len(df_cleaned)}, after: {len(df_no_outliers)}")
    
    st.markdown("---")
    
    # ==================== STEP 4: Feature Engineering ====================
    st.header("4️⃣ Feature Engineering")
    
    df_features = add_features(df_no_outliers.copy())
    
    # Show new features
    new_cols = [col for col in df_features.columns if col not in df_no_outliers.columns]
    st.success(f"Added {len(new_cols)} new features: {new_cols}")
    
    # Visualize new features
    st.subheader("Technical Indicators Visualization")
    
    if 'Close' in df_features.columns and 'MA_5' in df_features.columns:
        fig_tech, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Price with Moving Averages
        axes[0, 0].plot(df_features['Close'].tail(200), label='Close', alpha=0.8)
        axes[0, 0].plot(df_features['MA_5'].tail(200), label='MA_5', alpha=0.7)
        axes[0, 0].plot(df_features['MA_10'].tail(200), label='MA_10', alpha=0.7)
        axes[0, 0].plot(df_features['MA_20'].tail(200), label='MA_20', alpha=0.7)
        axes[0, 0].set_title('Price with Moving Averages')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MACD
        axes[0, 1].plot(df_features['MACD'].tail(200), label='MACD', color='blue')
        axes[0, 1].plot(df_features['MACD_Signal'].tail(200), label='Signal', color='red')
        axes[0, 1].set_title('MACD')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RSI
        axes[1, 0].plot(df_features['RSI'].tail(200), label='RSI', color='purple')
        axes[1, 0].axhline(y=70, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('RSI (Relative Strength Index)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bollinger Bands
        axes[1, 1].plot(df_features['Close'].tail(200), label='Close', alpha=0.8)
        axes[1, 1].plot(df_features['BB_Upper'].tail(200), label='Upper BB', color='red', alpha=0.7)
        axes[1, 1].plot(df_features['BB_Middle'].tail(200), label='Middle BB', color='orange', alpha=0.7)
        axes[1, 1].plot(df_features['BB_Lower'].tail(200), label='Lower BB', color='green', alpha=0.7)
        axes[1, 1].set_title('Bollinger Bands')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Volatility
        axes[2, 0].plot(df_features['Volatility'].tail(200), color='orange')
        axes[2, 0].set_title('Volatility')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Price Change
        axes[2, 1].plot(df_features['Price_Change'].tail(200), color='green', alpha=0.7)
        axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2, 1].set_title('Price Change %')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_tech)
    
    # Drop NaN rows created by feature engineering
    df_final = df_features.dropna()
    st.success(f"Final dataset shape after feature engineering: {df_final.shape}")
    
    st.markdown("---")
    
    # ==================== STEP 5: Model Training & Prediction ====================
    st.header("5️⃣ Model Training & Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Select Target Column", 
            df_final.columns.tolist(), index=df_final.columns.tolist().index('Close') if 'Close' in df_final.columns else 0)
    with col2:
        model_type = st.selectbox("Select Model", ['linear', 'random_forest', 'svm'], index=0)
    
    if st.button("🚀 Run Prediction", type="primary"):
        try:
            with st.spinner('Running prediction...'):
                y_train, y_pred_train, y_test, y_pred, accuracy, train_mse, train_r2, test_mse, test_r2 = run_prediction(
                    df_final, target_column, model_type
                )
            
            st.success("Prediction completed!")
            
            # ==================== Model Performance ====================
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy (within 5%)", f"{accuracy:.2f}%")
            with col2:
                st.metric("Training R²", f"{train_r2:.4f}")
            with col3:
                st.metric("Test R²", f"{test_r2:.4f}")
            with col4:
                st.metric("Test MSE", f"{test_mse:.4f}")
            
            # Performance metrics table
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy (%)', 'Training MSE', 'Training R²', 'Test MSE', 'Test R²'],
                'Value': [f"{accuracy:.2f}", f"{train_mse:.4f}", f"{train_r2:.4f}", f"{test_mse:.4f}", f"{test_r2:.4f}"]
            })
            st.table(metrics_df)
            
            st.markdown("---")
            
            # ==================== Predictions ====================
            st.subheader("📋 Predictions Table")
            
            pred_df = pd.DataFrame({
                'Actual': y_test.values.flatten() if hasattr(y_test, 'values') else y_test.flatten(),
                'Predicted': y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
            })
            st.dataframe(pred_df.head(20), use_container_width=True)
            
            # Download predictions
            csv = pred_df.to_csv(index=False)
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
            
            st.markdown("---")
            
            # ==================== Visualizations ====================
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Actual vs Predicted (Test Data)")
                fig1 = plot_actual_vs_predicted(y_test, y_pred)
                st.pyplot(fig1)
            
            with col2:
                st.write("### Scatter Plot (Training Data)")
                fig2 = plot_scatter(y_train, y_pred_train)
                st.pyplot(fig2)
            
            # Additional visualizations
            st.subheader("📉 Additional Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residual plot
                residuals = y_test.values - y_pred.flatten() if hasattr(y_test, 'values') else y_test - y_pred.flatten()
                fig_resid, ax = plt.subplots()
                ax.scatter(y_pred.flatten(), residuals, alpha=0.5)
                ax.axhline(y=0, color='red', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Plot')
                st.pyplot(fig_resid)
            
            with col2:
                # Prediction distribution
                fig_dist, ax = plt.subplots()
                ax.hist(y_test.values.flatten() if hasattr(y_test, 'values') else y_test, bins=30, alpha=0.5, label='Actual', color='blue')
                ax.hist(y_pred.flatten(), bins=30, alpha=0.5, label='Predicted', color='red')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.set_title('Actual vs Predicted Distribution')
                ax.legend()
                st.pyplot(fig_dist)
            
            # Time series comparison (if we have index)
            st.subheader("📊 Time Series Comparison")
            fig_ts, ax = plt.subplots(figsize=(14, 5))
            ax.plot(range(len(y_test)), y_test.values.flatten() if hasattr(y_test, 'values') else y_test, label='Actual', alpha=0.8)
            ax.plot(range(len(y_pred)), y_pred.flatten(), label='Predicted', alpha=0.8)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(target_column)
            ax.set_title(f'Actual vs Predicted Over Time ({target_column})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_ts)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👈 Please upload a CSV file or select sample data from the sidebar to begin!")
    
    # Show sample information
    st.markdown("""
    ### How to use:
    1. **Select Data Source**: Choose to upload your own CSV or use the TSLA sample data
    2. **Data Exploration**: View your data statistics and visualizations
    3. **Missing Values**: Choose a strategy to handle missing values
    4. **Outliers**: Detect and remove outliers using IQR or Z-score method
    5. **Feature Engineering**: Add technical indicators (MA, RSI, MACD, Bollinger Bands, etc.)
    6. **Run Prediction**: Select target column and model type, then run!
    """)

# Footer
st.markdown("---")
st.markdown("📈 **Stock Price Prediction Pipeline** | Built with Streamlit")

