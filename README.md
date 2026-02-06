# ⚡ Fed Funds Rate Prediction Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.0+-00D9FF.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-FF1744.svg)
![License](https://img.shields.io/badge/License-MIT-FFD700.svg)

**Premium Analytics Platform for Macroeconomic Forecasting**

*Lasso Regression with Real-Time FRED Data Integration*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API](#-api-configuration)

</div>

---

## 📊 Overview

A cutting-edge dashboard for predicting the **Federal Funds Rate** based on macroeconomic indicators. The system leverages **Lasso Regression** with time-lagged features and provides a professional web interface for interactive analysis.

### Core Capabilities

- 🎯 **Real-Time Data Retrieval** from the Federal Reserve Economic Data (FRED) API
- 📈 **Machine Learning Pipeline** with Scikit-Learn (Lasso, StandardScaler, TimeSeriesSplit)
- 🎨 **Premium Dark Theme** with neon accents and professional typography
- 📊 **4 Evaluation Plots**: KDE Distribution, Time Series, Scatter, Feature Importance
- ⚙️ **Interactive Hyperparameter Tuning** through modern web UI
- 💾 **SQLite Database** for efficient local data storage

---

## 🎨 Features

### Design & UX
- **Distinctive Typography**: JetBrains Mono for professional look
- **Premium Color Scheme**: Electric Cyan (#00F5FF), Neon Red (#FF1744), Gold (#FFD700)
- **Gradient Backgrounds**: Animated gradients and transparencies
- **Custom Scrollbar**: Gradient-styled for consistent design
- **Responsive Layout**: 2-column grid system without overlapping
- **Loading Indicators**: Smooth transitions during computations

### Analytics
- **KDE Distribution Plot**: Comparison of Actual vs. Predicted distributions
- **Time Series Forecast**: Time series visualization with confidence intervals
- **Prediction Scatter**: Correlation between predictions and reality
- **Feature Importance**: Bar chart of Lasso coefficients

### Model Performance Metrics
- **R² Score**: Coefficient of determination for model quality
- **R² CV (Cross-Validation)**: 10-Fold Time Series Split median
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
pip (Python Package Manager)
FRED API Key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
```

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/fed-funds-predictor.git
cd fed-funds-predictor
```

### 2. Create Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install dash plotly pandas numpy scikit-learn fredapi scipy
```

### 4. Create Project Structure

```bash
mkdir data
```

### 5. Create Configuration File

Create `data/configs.json`:

```json
{
  "FRED_API_KEY": "your_api_key_here"
}
```

> 💡 **Get API Key**: Register for free at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## 🎮 Usage

### Start Dashboard

```bash
python fed_dashboard_premium.py
```

The dashboard will be accessible at:
```
http://localhost:8050
```

### Workflow

1. **Feature Selection**: Choose macroeconomic indicators from dropdown
2. **Data Configuration**: Set start date and feature lag (months)
3. **Hyperparameter Tuning**: 
   - **Alpha**: Regularization strength (0.01 - 0.10)
   - **Test Size**: Train/Test split ratio (0.10 - 0.50)
4. **Run Evaluation**: Click "🚀 RUN MODEL EVALUATION"
5. **Analyze Results**: Interpret the 4 visualizations and metrics

### Example Configuration

```
Features: CPIAUCSL, UNRATE, M2SL
Start Date: 2019-01-01
Feature Lag: 4 months
Alpha: 0.03
Test Size: 0.25
```

---

## 🏗️ Architecture

### Data Flow

```
FRED API → SQLite Database → Pandas DataFrame → Preprocessing → Model Training → Evaluation Plots
```

### Components

#### 1. **Data Management** (`Config`, `initialize_fred_api`, `setup_database`)
- Loads FRED API key from configuration
- Retrieves 8 macroeconomic time series
- Stores in local SQLite database (`macrodata.db`)

#### 2. **Feature Engineering**
- **Time Lag**: Features are lagged by n months
- **Percentage Change**: Monthly change rates (%)
- **Rolling Mean**: 3-month average for smoothing
- **Standardization**: Z-score normalization for Lasso

#### 3. **Model Pipeline**
```python
StandardScaler → Train/Test Split → Lasso(alpha) → Cross-Validation → Predictions
```

#### 4. **Visualization** (`create_evaluation_plots`)
- Plotly subplots with dark theme
- Custom hover templates
- Gradient-based color schemes
- Interactive legends and zoom

---

## 📊 Macroeconomic Indicators

| FRED ID   | Indicator                        | Description                                     |
|-----------|----------------------------------|-------------------------------------------------|
| CPIAUCSL  | Consumer Price Index             | Inflation measure based on consumer prices     |
| PPIACO    | Producer Price Index             | Wholesale price index                          |
| UNRATE    | Unemployment Rate                | Unemployment rate (%)                          |
| PAYEMS    | Total Nonfarm Payrolls           | Employment outside agriculture                 |
| PCEPILFE  | Core PCE Price Index             | Core personal consumption expenditures rate    |
| M2SL      | M2 Money Stock                   | M2 money supply                                |
| FEDFUNDS  | Federal Funds Rate               | **Target Variable** - Fed's key interest rate  |
| UMCSENT   | Consumer Sentiment               | Consumer confidence (University of Michigan)   |

---

## ⚙️ API Configuration

### FRED API Limits

- **Rate Limit**: 120 requests per minute
- **Data History**: Depends on series (mostly from 1950s)
- **Update Frequency**: Daily, weekly, or monthly
- **Format**: JSON, XML, Text

### Adding Custom Indicators

Edit the `Config` class in `fed_dashboard_premium.py`:

```python
class Config:
    FEATURE_IDS = [
        "CPIAUCSL",
        "PPIACO",
        # ... existing features
        "DEXUSEU",  # Example: USD/EUR Exchange Rate
        "DGS10"     # Example: 10-Year Treasury Rate
    ]
    
    FEATURE_LABELS = {
        "DEXUSEU": "USD/EUR Exchange Rate",
        "DGS10": "10-Year Treasury Yield"
    }
```

---

## 🔬 Model Details

### Lasso Regression

**Why Lasso?**
- ✅ **Feature Selection**: L1 regularization sets unimportant coefficients to zero
- ✅ **Multicollinearity**: Robust against correlated features
- ✅ **Interpretability**: Clear weighting of indicators
- ✅ **Overfitting Prevention**: Regularization prevents overly complex models

### Hyperparameters

**Alpha (Regularization Strength)**
- **Range**: 0.01 - 0.10
- **Default**: 0.03
- **Effect**: Higher values → more sparsity (fewer features)

**Test Size**
- **Range**: 0.10 - 0.50
- **Default**: 0.25
- **Effect**: Larger test sets → more robust evaluation

### Cross-Validation

**TimeSeriesSplit** with 10 folds:
- Respects temporal order of data
- Prevents data leakage
- More realistic performance estimation

---

## 📈 Interpreting the Plots

### 1. Distribution Comparison (KDE)
- **Cyan**: Actual Fed Funds Rate distribution
- **Red**: Predicted distribution
- **Goal**: Maximum overlap between curves

### 2. Time Series Forecast
- **Cyan Line**: Actual values (test set)
- **Red Dashed**: Model predictions
- **Goal**: Minimal deviation between lines

### 3. Prediction Scatter
- **Perfect Prediction Line**: Diagonal (gold dashed)
- **Points**: Each point = one prediction
- **Color**: Plasma colorscale shows actual value
- **Goal**: Points close to diagonal

### 4. Feature Importance
- **Gold Bar**: Most important indicator
- **Red Bars**: Other features by coefficient magnitude
- **Goal**: Understand which indicators matter most

---

## 🐛 Troubleshooting

### Problem: API Key Error
```
Error: Invalid API key
```
**Solution**: Check `data/configs.json` - API key must be valid

### Problem: Database Locked
```
sqlite3.OperationalError: database is locked
```
**Solution**: Close other connections to DB or delete `data/macrodata.db`

### Problem: Import Error
```
ModuleNotFoundError: No module named 'fredapi'
```
**Solution**: `pip install fredapi`

### Problem: Empty Plots
```
Empty figure returned
```
**Solution**: Click the "RUN MODEL EVALUATION" button

### Problem: Empty Feature Dropdown
```
No features available
```
**Solution**: Check internet connection - FRED API must be reachable

---

## 📚 Technology Stack

| Component          | Technology         | Version |
|--------------------|--------------------|---------|
| **Backend**        | Python             | 3.8+    |
| **Web Framework**  | Dash               | 2.0+    |
| **Plotting**       | Plotly             | 5.0+    |
| **ML Library**     | Scikit-Learn       | 1.0+    |
| **Data API**       | fredapi            | 0.5+    |
| **Database**       | SQLite             | 3.0+    |
| **Data Processing**| Pandas, NumPy      | Latest  |
| **Statistics**     | SciPy              | Latest  |

---

## 🎓 Scientific Background

### Why Time-Lagged Features?

Economic indicators typically have a **delayed effect** on monetary policy. For example:
- Rising CPI (inflation) → Fed raises rates 2-6 months later
- Increasing unemployment → Fed lowers rates with delay

By lagging features, the model learns these **temporal relationships**.

### Feature Transformation Pipeline

```
Raw FRED Data (levels)
    ↓
Percentage Change (month-over-month growth rates)
    ↓
Rolling Mean (3-month smoothing to reduce noise)
    ↓
Time Lag (shift features by n months)
    ↓
Standardization (Z-score normalization)
    ↓
Lasso Regression
```

### Model Assumptions

1. **Linearity**: Fed Funds Rate responds linearly to economic indicators
2. **Stationarity**: Using percentage changes makes series stationary
3. **No Multicollinearity**: Lasso handles correlated features well
4. **Temporal Causality**: Past indicators predict future rates

---

## 📊 Performance Benchmarks

### Typical Results (2019-2024 data)

```
R² Score:          0.75 - 0.85
R² CV (Median):    0.70 - 0.80
RMSE:              0.30 - 0.50 percentage points
MAE:               0.20 - 0.40 percentage points
```

### Interpretation

- **R² > 0.75**: Model explains 75%+ of variance → Good fit
- **RMSE < 0.5**: Predictions within ±0.5% on average → Acceptable
- **R² CV ≈ R²**: Low overfitting → Generalizes well

---

## 🔒 Data Privacy & Security

- **API Key**: Stored locally in `configs.json` - **never commit to Git**
- **Database**: Local SQLite file - no cloud storage
- **FRED Data**: Publicly available, no sensitive information
- **No User Tracking**: Dashboard runs entirely locally

### Best Practices

1. Add `data/configs.json` to `.gitignore`
2. Use environment variables for production deployments
3. Rotate API keys periodically
4. Keep SQLite database file private

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features
- Update README with new features

---

## 📖 References

### Academic Papers
- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso"
- Taylor, J. B. (1993). "Discretion versus Policy Rules in Practice"
- Bernanke, B. S. & Blinder, A. S. (1992). "The Federal Funds Rate and the Channels of Monetary Transmission"

### Data Sources
- [FRED - Federal Reserve Economic Data](https://fred.stlouisfed.org/)
- [Federal Reserve Board](https://www.federalreserve.gov/)

### Documentation
- [Scikit-Learn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Dash Framework](https://dash.plotly.com/)
- [Plotly Python](https://plotly.com/python/)

---

## 📄 License

MIT License

```
Copyright (c) 2026 Marlon Elias Braje

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **FRED API**: Federal Reserve Bank of St. Louis for providing free economic data
- **Dash Team**: Plotly Technologies for the excellent web framework
- **Scikit-Learn**: For robust machine learning implementations
- **Design Inspiration**: Modern financial analytics dashboards

---

## 📧 Contact

For questions or feedback:
- 📧 Email: marlonbj@outlook.de
- 🐙 GitHub: [marlonbje](https://github.com/marlonbje)
- 💼 LinkedIn: [Marlon Elias Braje](https://www.linkedin.com/in/marlon-elias-braje-5534b8344)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Made with ⚡ by Marlon Elias Braje**

*Powered by Machine Learning & Federal Reserve Data*

[![GitHub](https://img.shields.io/github/stars/yourusername/fed-funds-predictor?style=social)](https://github.com/yourusername/fed-funds-predictor)

</div>
