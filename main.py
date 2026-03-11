"""
Tesla Stock Price Prediction - Full Data Preprocessing Pipeline
With: Missing Value Handling, Outlier Removal, Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


def load_data(filepath):
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    """
    Explore the dataset - display info, statistics, and basic visualizations.
    
    Args:
        df: DataFrame to explore
        
    Returns:
        Dictionary with exploration results
    """
    results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes,
        'missing_values': df.isnull().sum(),
        'describe': df.describe()
    }
    
    # Print basic info
    print("=" * 50)
    print("DATA EXPLORATION")
    print("=" * 50)
    print(f"\nShape: {results['shape']}")
    print(f"\nColumns: {results['columns']}")
    print(f"\nData Types:\n{results['dtypes']}")
    print(f"\nMissing Values:\n{results['missing_values']}")
    print(f"\nStatistical Summary:\n{results['describe']}")
    
    return results


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame to process
        strategy: 'mean', 'median', 'drop', or 'forward_fill'
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    missing_count = df.isnull().sum().sum()
    
    if missing_count == 0:
        print("No missing values found.")
        return df
    
    print(f"Found {missing_count} missing values.")
    
    if strategy == 'mean':
        # Fill numeric columns with mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print("Missing values filled with mean.")
        
    elif strategy == 'median':
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print("Missing values filled with median.")
        
    elif strategy == 'drop':
        # Drop rows with missing values
        df = df.dropna()
        print("Rows with missing values dropped.")
        
    elif strategy == 'forward_fill':
        # Forward fill missing values
        df = df.fillna(method='ffill')
        print("Missing values forward filled.")
    
    return df


def handle_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect and remove outliers from the dataset.
    
    Args:
        df: DataFrame to process
        column: Column name to check for outliers
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for zscore)
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    initial_len = len(df)
    
    if method == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"IQR Outlier Removal: {initial_len - len(df)} rows removed from '{column}'")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
    elif method == 'zscore':
        # Z-score method
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column]))
        df = df[z_scores < threshold]
        print(f"Z-score Outlier Removal: {initial_len - len(df)} rows removed from '{column}'")
    
    return df


def add_features(df):
    """
    Add technical indicators as features for stock prediction.
    
    Args:
        df: DataFrame with stock data
        
    Returns:
        DataFrame with added features
    """
    df = df.copy()
    
    # Moving Averages
    if 'Close' in df.columns:
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Volume features
        if 'Volume' in df.columns:
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_Change'] = df['Volume'].pct_change()
        
        print(f"Added {12 if 'Volume' in df.columns else 10} new features.")
    
    return df


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess data - scale features and split into train/test sets.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_model(model_type):
    """
    Get the specified model.
    
    Args:
        model_type: 'linear', 'random_forest', or 'svm'
        
    Returns:
        Model instance
    """
    if model_type == 'linear':
        return LinearRegression()
    elif model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        return SVR(kernel='rbf')
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_prediction(df, target_column='Close', model_type='linear'):
    """
    Run the full prediction pipeline.
    
    Args:
        df: DataFrame with stock data
        target_column: Column to predict
        model_type: Type of model to use ('linear', 'random_forest', 'svm')
        
    Returns:
        y_train, y_pred_train, y_test, y_pred, accuracy, train_mse, train_r2, test_mse, test_r2
    """
    # Make a copy
    df = df.copy()
    
    # Handle missing values
    df = handle_missing_values(df, strategy='forward_fill')
    
    # Drop any remaining NaN rows after feature engineering
    df = df.dropna()
    
    # Prepare features and target
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")
    
    # Use numeric columns as features (exclude target)
    feature_cols = [col for col in df.columns if col != target_column and df[col].dtype in [np.float64, np.int64]]
    X = df[feature_cols]
    y = df[target_column]
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Get and train model
    model = get_model(model_type)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (percentage of predictions within 5% of actual)
    accuracy = 100 * np.mean(np.abs((y_test - y_pred) / y_test) < 0.05)
    
    print(f"\nModel Performance ({model_type}):")
    print(f"Training MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
    print(f"Accuracy (within 5%): {accuracy:.2f}%")
    
    return y_train, y_pred_train, y_test, y_pred, accuracy, train_mse, train_r2, test_mse, test_r2


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot actual vs predicted values for test data.
    
    Args:
        y_test: Actual target values
        y_pred: Predicted values
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolors='black')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Actual vs Predicted (Test Data)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_scatter(y_train, y_pred_train):
    """
    Plot scatter plot for training data.
    
    Args:
        y_train: Actual target values for training
        y_pred_train: Predicted values for training
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_train, y_pred_train, alpha=0.5, color='green', edgecolors='black')
    
    # Perfect prediction line
    min_val = min(y_train.min(), y_pred_train.min())
    max_val = max(y_train.max(), y_pred_train.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Scatter Plot (Training Data)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def full_pipeline(filepath, target_column='Close', model_type='linear'):
    """
    Run the full data processing and prediction pipeline.
    
    Args:
        filepath: Path to the CSV file
        target_column: Column to predict
        model_type: Type of model to use
        
    Returns:
        Dictionary with results
    """
    # Load data
    print("Loading data...")
    df = load_data(filepath)
    
    # Explore data
    print("\nExploring data...")
    explore_data(df)
    
    # Handle missing values
    print("\nHandling missing values...")
    df = handle_missing_values(df, strategy='forward_fill')
    
    # Handle outliers in Close price
    if target_column in df.columns:
        print(f"\nHandling outliers in '{target_column}'...")
        df = handle_outliers(df, target_column, method='iqr', threshold=1.5)
    
    # Add features
    print("\nAdding features...")
    df = add_features(df)
    
    # Drop NaN rows created by feature engineering
    df = df.dropna()
    
    # Run prediction
    print("\nRunning prediction...")
    results = run_prediction(df, target_column, model_type)
    
    return results


# Main execution
if __name__ == "__main__":
    # Example usage
    print("Stock Price Prediction Pipeline")
    print("=" * 50)
    
    # Test with sample data
    try:
        results = full_pipeline('../Datasets/TSLA.csv', target_column='Close', model_type='linear')
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"Error: {e}")

