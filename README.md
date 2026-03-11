# 📈 Stock Price Prediction Pipeline

A comprehensive machine learning application for predicting stock prices using various ML algorithms with full data preprocessing pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit-learn-1.3+-green.svg)

---

## 🎯 Project Overview

This project is a **Stock Price Prediction Pipeline** that allows users to:

- 📊 **Explore Data**: Analyze stock data with statistics and visualizations
- 🧹 **Clean Data**: Handle missing values and remove outliers
- ⚙️ **Feature Engineering**: Add technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- 🤖 **Predict Prices**: Use Linear Regression, Random Forest, or SVM models
- 📈 **Visualize Results**: View actual vs predicted plots and performance metrics

---

## 📁 Project Structure

```
Stock_Price_Predication/
├── main.py                 # Core ML pipeline functions
├── streamlit_app.py        # Web interface
├── requirements.txt        # Dependencies
├── run_app.bat            # Quick start script (Windows)
├── README.md              # This file
└── Datasets/             # Sample data directory
    └── TSLA.csv          # Tesla stock data
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit |
| **ML Framework** | scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |

---

## 📋 Features

### 1. Data Exploration
- View dataset shape, columns, and data types
- Statistical summary (mean, std, min, max, etc.)
- Missing values analysis with visualization

### 2. Missing Values Handling
- **Forward Fill**: Use previous value to fill gaps
- **Mean**: Fill with column average
- **Median**: Fill with middle value
- **Drop**: Remove rows with missing data

### 3. Outlier Detection & Removal
- **IQR Method**: Interquartile Range detection
- **Z-Score Method**: Standard deviation-based detection
- Visualize outliers with box plots and histograms

### 4. Feature Engineering
Add technical indicators:
| Indicator | Description |
|-----------|-------------|
| **MA_5, MA_10, MA_20** | Moving Averages (5, 10, 20 days) |
| **EMA_12, EMA_26** | Exponential Moving Averages |
| **MACD** | Moving Average Convergence Divergence |
| **RSI** | Relative Strength Index |
| **Bollinger Bands** | Price volatility bands |
| **Volatility** | Standard deviation of prices |
| **Price Change** | Daily percentage change |

### 5. Model Training & Prediction
- **Linear Regression**: Simple linear model
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Regression

### 6. Visualizations
- Actual vs Predicted scatter plots
- Residual analysis
- Prediction distribution
- Time series comparison
- Technical indicator charts

---

## 🚀 How to Run

### Option 1: Quick Start (Windows)
```bash
# Double-click run_app.bat
```

### Option 2: Manual Setup

#### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd Stock_Price_Predication
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv

# Activate
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the App
```bash
streamlit run streamlit_app.py
```

#### 5. Open Browser
Navigate to: **http://localhost:8501**

---

## 📖 How to Use the App

### Step 1: Select Data Source
- Choose **"Upload CSV"** to use your own stock data
- Or select **"Use Sample Data (TSLA)"** for demo

### Step 2: Explore Your Data
- View dataset statistics and missing values
- Understand your data distribution

### Step 3: Handle Missing Values
- Select a strategy from the dropdown
- Click to apply

### Step 4: Remove Outliers
- Choose column (usually 'Close')
- Select method (IQR or Z-Score)
- Adjust threshold if needed
- View box plot and histogram

### Step 5: Feature Engineering
- Automatically adds technical indicators
- View charts for MA, RSI, MACD, Bollinger Bands

### Step 6: Run Prediction
- Select target column (e.g., 'Close')
- Choose model (Linear, Random Forest, SVM)
- Click **"Run Prediction"**

### Step 7: View Results
- Model performance metrics
- Predictions table with download option
- Various visualization plots
- Download your predictions as CSV

---

## 📊 Sample Output

```
Data Exploration:
- Shape: (2956, 6)
- Columns: Open, High, Low, Close, Adj Close, Volume
- Missing Values: 0

After Outlier Removal:
- Rows: 2445 (removed 511 outliers)

After Feature Engineering:
- New Features: 12 (MA, RSI, MACD, etc.)

Model Performance (Linear):
- Accuracy: 100.00%
- Training R²: 1.0000
- Test R²: 1.0000
```

---

## 🔧 Customization

### Add Your Own Data
1. Prepare a CSV file with columns: Open, High, Low, Close, Volume
2. Upload in the app or place in `Datasets/` folder

### Modify Models
Edit `main.py` to:
- Add new algorithms (XGBoost, LSTM, etc.)
- Tune hyperparameters
- Add custom features

### Change Visualizations
Edit `streamlit_app.py` to:
- Add new plots
- Change color schemes
- Add more metrics

---

## 📝 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

## 🎓 Understanding the Pipeline

```
Raw Data → Missing Values → Outliers → Features → Model → Predictions
    ↓           ↓              ↓          ↓          ↓         ↓
  CSV File   Handle NaN    IQR/Z-Score  TA Indicators  ML Algo   Results
```

### Why This Pipeline?

1. **Missing Values**: Real data often has gaps
2. **Outliers**: Extreme prices can skew predictions
3. **Features**: Technical indicators improve accuracy
4. **Multiple Models**: Compare different algorithms

---

## 🔗 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Technical Indicators Guide](https://www.investopedia.com/terms/t/technicalindicator.asp)

---

## 📄 License

This project is for educational purposes.

---

## 👨‍💻 Author

Created for Machine Learning practical implementation.

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Run with `streamlit run streamlit_app.py --server.port 8502` |
| Missing modules | Run `pip install -r requirements.txt` |
| CSV format error | Ensure CSV has headers: Open, High, Low, Close, Volume |
| Memory issues | Use smaller dataset or reduce features |

---

**Made with ❤️ using Streamlit and scikit-learn**

