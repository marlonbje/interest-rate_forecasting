"""
Macroeconomic Fed Funds Rate Prediction Dashboard
Premium Analytics Platform with Professional Design
"""

import json
import sqlite3
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output, State
from fredapi import Fred
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler


# ==================== Configuration ====================
class Config:
    """Application configuration and FRED feature IDs."""
    FEATURE_IDS = [
        "CPIAUCSL",   # Consumer Price Index
        "PPIACO",     # Producer Price Index
        "UNRATE",     # Unemployment Rate
        "PAYEMS",     # Total Nonfarm Payrolls
        "PCEPILFE",   # Core PCE Price Index
        "M2SL",       # M2 Money Stock
        "FEDFUNDS",   # Federal Funds Rate (Target)
        "UMCSENT"     # Consumer Sentiment
    ]
    
    FEATURE_LABELS = {
        "CPIAUCSL": "Consumer Price Index",
        "PPIACO": "Producer Price Index",
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls",
        "PCEPILFE": "Core PCE Inflation",
        "M2SL": "M2 Money Supply",
        "UMCSENT": "Consumer Sentiment"
    }
    
    DB_PATH = "data/macrodata.db"
    CONFIG_PATH = "data/configs.json"


# ==================== Data Management ====================
def initialize_fred_api() -> Fred:
    """Load API key and initialize FRED API client."""
    with open(Config.CONFIG_PATH, "r") as file:
        api_key = json.load(file)["FRED_API_KEY"]
    return Fred(api_key)


def setup_database(fred: Fred) -> sqlite3.Connection:
    """Download FRED data and populate SQLite database."""
    con = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
    
    for feature_id in Config.FEATURE_IDS:
        series = fred.get_series(feature_id).rename(feature_id)
        series.to_sql(
            name=feature_id,
            con=con,
            if_exists="replace",
            index_label="Datetime"
        )
    
    return con


def get_available_features(con: sqlite3.Connection) -> List[str]:
    """Retrieve list of available feature tables (excluding target variable)."""
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [name[0] for name in cur.fetchall()]
    
    if "FEDFUNDS" in tables:
        tables.remove("FEDFUNDS")
    
    return tables


# ==================== Model Evaluation ====================
def evaluate_model(
    df: pd.DataFrame,
    offset: int,
    test_size: float,
    alpha: float
) -> go.Figure:
    """
    Train Lasso model and create comprehensive evaluation plots.
    
    Args:
        df: DataFrame with features and target
        offset: Number of periods to lag features
        test_size: Proportion of data for testing
        alpha: Lasso regularization parameter
        
    Returns:
        Plotly figure with 4 evaluation subplots
    """
    df = df.dropna()
    
    # Prepare lagged features
    shifted_features = df.drop("FEDFUNDS", axis=1).shift(int(offset)).dropna()
    valid_idx = shifted_features.index
    
    # Scale features and split data
    scaler = StandardScaler()
    X = scaler.fit_transform(shifted_features)
    y = np.array(df.FEDFUNDS.loc[valid_idx])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=25, test_size=float(test_size)
    )
    
    # Train model
    model = Lasso(
        alpha=float(alpha),
        fit_intercept=True,
        max_iter=1200,
        warm_start=True,
        random_state=25
    )
    model.fit(X=X_train, y=y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test).flatten()
    y_test = y_test.flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(model, X_train, y_train, y_test, y_pred)
    
    # Create visualization
    fig = create_evaluation_plots(
        y_test, y_pred, shifted_features.columns, model.coef_, metrics
    )
    
    return fig


def calculate_metrics(
    model: Lasso,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate model performance metrics."""
    cv_scores = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=TimeSeriesSplit(n_splits=10),
        scoring='r2'
    )
    
    metrics = {
        "R²": r2_score(y_true=y_test, y_pred=y_pred),
        "R² CV": np.median(cv_scores),
        "RMSE": np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)),
        "MAE": mean_absolute_error(y_true=y_test, y_pred=y_pred)
    }
    
    # Print metrics to console
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    for key, val in metrics.items():
        print(f"{key:20s}: {val:.4f}")
    print("="*60 + "\n")
    
    return metrics


def create_evaluation_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    coefficients: np.ndarray,
    metrics: Dict[str, float]
) -> go.Figure:
    """Create professional evaluation plots with premium dark theme."""
    
    # Create subplot structure with proper spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Distribution Comparison (KDE)</b>",
            "<b>Time Series: Actual vs Predicted</b>",
            "<b>Prediction Scatter Plot</b>",
            "<b>Feature Importance (|Coefficients|)</b>"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.14,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # Premium color scheme - Neon on Dark
    COLOR_ACTUAL = "#00F5FF"      # Electric Cyan
    COLOR_PRED = "#FF1744"        # Neon Red
    COLOR_ACCENT = "#FFD700"      # Gold
    COLOR_GRID = "#1a1a1a"
    COLOR_TEXT = "#E0E0E0"
    
    # 1. KDE Distribution Plot
    y_test_kde = gaussian_kde(y_test)
    y_pred_kde = gaussian_kde(y_pred)
    mean, std = np.mean(y_test), np.std(y_test)
    x_range = np.linspace(mean - 3*std, mean + 3*std, 1000)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_test_kde(x_range),
            mode="lines",
            name="Actual Distribution",
            line=dict(color=COLOR_ACTUAL, width=3),
            fill='tozeroy',
            fillcolor=f'rgba(0, 245, 255, 0.15)',
            hovertemplate='<b>Rate:</b> %{x:.2f}%<br><b>Density:</b> %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred_kde(x_range),
            mode="lines",
            name="Predicted Distribution",
            line=dict(color=COLOR_PRED, width=3, dash='dot'),
            fill='tozeroy',
            fillcolor=f'rgba(255, 23, 68, 0.15)',
            hovertemplate='<b>Rate:</b> %{x:.2f}%<br><b>Density:</b> %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Time Series Plot
    x_axis = np.arange(len(y_test))
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_test,
            name="Actual Values",
            mode="lines+markers",
            line=dict(color=COLOR_ACTUAL, width=2.5),
            marker=dict(size=5, symbol='circle', line=dict(width=1, color='white')),
            hovertemplate='<b>Index:</b> %{x}<br><b>Actual Rate:</b> %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_pred,
            name="Predictions",
            mode="lines+markers",
            line=dict(color=COLOR_PRED, width=2.5, dash='dash'),
            marker=dict(size=5, symbol='diamond', line=dict(width=1, color='white')),
            hovertemplate='<b>Index:</b> %{x}<br><b>Predicted Rate:</b> %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Scatter Plot with perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=y_test,
            mode="markers",
            name="Predictions",
            marker=dict(
                size=10,
                color=y_test,
                colorscale='Plasma',
                showscale=True,
                line=dict(width=1.5, color='rgba(255,255,255,0.6)'),
                colorbar=dict(
                    title=dict(text="<b>Actual<br>Rate (%)</b>", font=dict(size=11)),
                    x=0.46,
                    len=0.35,
                    thickness=12,
                    tickfont=dict(size=10, color=COLOR_TEXT),
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='rgba(255,255,255,0.3)',
                    borderwidth=1
                )
            ),
            showlegend=False,
            hovertemplate='<b>Predicted:</b> %{x:.2f}%<br><b>Actual:</b> %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min()) - 0.5
    max_val = max(y_test.max(), y_pred.max()) + 0.5
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color=COLOR_ACCENT, width=2, dash="dot"),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # 4. Feature Importance Bar Chart
    abs_coefs = np.abs(coefficients)
    colors_bars = [COLOR_ACCENT if c == abs_coefs.max() else COLOR_PRED for c in abs_coefs]
    
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=abs_coefs,
            name="Coefficient Magnitude",
            marker=dict(
                color=colors_bars,
                line=dict(color='rgba(255,255,255,0.4)', width=1.5),
                pattern=dict(shape="/", bgcolor="rgba(0,0,0,0.3)", size=4, solidity=0.3)
            ),
            showlegend=False,
            hovertemplate='<b>%{x}</b><br><b>|Coefficient|:</b> %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update all axes with consistent styling
    axis_style = dict(
        showgrid=True,
        gridcolor=COLOR_GRID,
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.2)',
        zerolinewidth=2,
        color=COLOR_TEXT,
        tickfont=dict(size=10, family='JetBrains Mono, Consolas, monospace'),
        titlefont=dict(size=12, family='JetBrains Mono, Consolas, monospace', color='white')
    )
    
    # Apply axis labels and styling
    fig.update_xaxes(title_text="<b>Fed Funds Rate (%)</b>", row=1, col=1, **axis_style)
    fig.update_yaxes(title_text="<b>Probability Density</b>", row=1, col=1, **axis_style)
    
    fig.update_xaxes(title_text="<b>Test Sample Index</b>", row=1, col=2, **axis_style)
    fig.update_yaxes(title_text="<b>Fed Funds Rate (%)</b>", row=1, col=2, **axis_style)
    
    fig.update_xaxes(title_text="<b>Predicted Rate (%)</b>", row=2, col=1, **axis_style)
    fig.update_yaxes(title_text="<b>Actual Rate (%)</b>", row=2, col=1, **axis_style)
    
    fig.update_xaxes(title_text="<b>Economic Indicator</b>", row=2, col=2, **axis_style, tickangle=-35)
    fig.update_yaxes(title_text="<b>|Coefficient|</b>", row=2, col=2, **axis_style)
    
    # Overall layout with premium dark theme
    fig.update_layout(
        height=900,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(
            color='white',
            size=11,
            family='JetBrains Mono, Consolas, monospace'
        ),
        title=dict(
            text=(
                f"<b style='font-size:24px; color:#00F5FF;'>FED FUNDS RATE PREDICTION MODEL</b><br>"
                f"<span style='font-size:14px; color:#888;'>Lasso Regression with Macroeconomic Indicators</span><br>"
                f"<span style='font-size:13px; color:#FFD700;'>"
                f"R² = {metrics['R²']:.4f} │ "
                f"R² CV = {metrics['R² CV']:.4f} │ "
                f"RMSE = {metrics['RMSE']:.4f} │ "
                f"MAE = {metrics['MAE']:.4f}"
                f"</span>"
            ),
            font=dict(family='JetBrains Mono, Consolas, monospace'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26,26,26,0.9)',
            bordercolor='rgba(0, 245, 255, 0.5)',
            borderwidth=2,
            font=dict(size=11, family='JetBrains Mono, Consolas, monospace'),
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top'
        ),
        margin=dict(t=160, b=80, l=80, r=80),
        hovermode='closest'
    )
    
    # Update subplot titles styling
    for annotation in fig['layout']['annotations'][:4]:
        annotation['font'] = dict(
            size=13,
            color='#00F5FF',
            family='JetBrains Mono, Consolas, monospace'
        )
        annotation['y'] = annotation['y'] + 0.01
    
    return fig


# ==================== Dash Application ====================
def create_app(con: sqlite3.Connection, available_features: List[str]) -> Dash:
    """Create and configure Dash application with premium dark theme."""
    
    app = Dash(__name__)
    
    # External CSS for premium fonts
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Fed Funds Rate Predictor</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    background: linear-gradient(135deg, #000000 0%, #0a0a0a 50%, #000000 100%);
                    background-attachment: fixed;
                    min-height: 100vh;
                }
                
                /* Animated gradient background effect */
                @keyframes gradient {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                
                /* Custom scrollbar */
                ::-webkit-scrollbar {
                    width: 10px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #0a0a0a;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: linear-gradient(180deg, #00F5FF, #FF1744);
                    border-radius: 5px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(180deg, #00D9FF, #FF0033);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    app.layout = html.Div(
        style={
            'backgroundColor': 'transparent',
            'color': 'white',
            'fontFamily': 'JetBrains Mono, Consolas, monospace',
            'padding': '40px 60px',
            'maxWidth': '1600px',
            'margin': '0 auto'
        },
        children=[
            # Header Section with Gradient Border
            html.Div(
                style={
                    'background': 'linear-gradient(135deg, rgba(0,245,255,0.1) 0%, rgba(255,23,68,0.1) 100%)',
                    'border': '2px solid',
                    'borderImage': 'linear-gradient(135deg, #00F5FF, #FF1744) 1',
                    'borderRadius': '12px',
                    'padding': '30px',
                    'marginBottom': '40px',
                    'boxShadow': '0 0 40px rgba(0,245,255,0.2)',
                    'animation': 'fadeIn 0.8s ease-out'
                },
                children=[
                    html.H1(
                        "⚡ FED FUNDS RATE PREDICTOR",
                        style={
                            'textAlign': 'center',
                            'marginBottom': '10px',
                            'background': 'linear-gradient(135deg, #00F5FF 0%, #FFD700 50%, #FF1744 100%)',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent',
                            'fontSize': '42px',
                            'fontWeight': '700',
                            'letterSpacing': '2px',
                            'textTransform': 'uppercase'
                        }
                    ),
                    html.P(
                        "Advanced Lasso Regression with Macroeconomic Time Series Analysis",
                        style={
                            'textAlign': 'center',
                            'color': '#888',
                            'fontSize': '14px',
                            'letterSpacing': '1px',
                            'fontWeight': '400'
                        }
                    ),
                ]
            ),
            
            # Control Panel Section
            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '30px',
                    'marginBottom': '40px'
                },
                children=[
                    # Left Column - Feature Selection & Date
                    html.Div(
                        style={
                            'background': 'rgba(26,26,26,0.8)',
                            'border': '1px solid rgba(0,245,255,0.3)',
                            'borderRadius': '10px',
                            'padding': '25px',
                            'boxShadow': '0 4px 20px rgba(0,0,0,0.5)'
                        },
                        children=[
                            html.H3(
                                "📊 DATA CONFIGURATION",
                                style={
                                    'color': '#00F5FF',
                                    'marginBottom': '20px',
                                    'fontSize': '16px',
                                    'fontWeight': '700',
                                    'letterSpacing': '1px',
                                    'borderBottom': '2px solid rgba(0,245,255,0.3)',
                                    'paddingBottom': '10px'
                                }
                            ),
                            
                            html.Label(
                                "Select Economic Indicators:",
                                style={
                                    'fontWeight': '500',
                                    'marginBottom': '8px',
                                    'display': 'block',
                                    'color': '#E0E0E0',
                                    'fontSize': '13px'
                                }
                            ),
                            dcc.Dropdown(
                                id="features",
                                options=[
                                    {'label': f"{feat} - {Config.FEATURE_LABELS.get(feat, feat)}", 
                                     'value': feat} 
                                    for feat in available_features
                                ],
                                value=available_features[:3],
                                multi=True,
                                style={
                                    'marginBottom': '20px',
                                },
                                className='custom-dropdown'
                            ),
                            
                            html.Div(
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': '1fr 1fr',
                                    'gap': '15px'
                                },
                                children=[
                                    html.Div([
                                        html.Label(
                                            "Start Date:",
                                            style={
                                                'fontWeight': '500',
                                                'marginBottom': '8px',
                                                'display': 'block',
                                                'color': '#E0E0E0',
                                                'fontSize': '13px'
                                            }
                                        ),
                                        dcc.Input(
                                            id="date",
                                            type="text",
                                            value="2019-01-01",
                                            placeholder="YYYY-MM-DD",
                                            style={
                                                'backgroundColor': '#0a0a0a',
                                                'color': '#00F5FF',
                                                'border': '1px solid rgba(0,245,255,0.4)',
                                                'borderRadius': '6px',
                                                'padding': '10px',
                                                'width': '100%',
                                                'fontSize': '13px',
                                                'fontFamily': 'JetBrains Mono, monospace',
                                                'fontWeight': '500'
                                            }
                                        ),
                                    ]),
                                    
                                    html.Div([
                                        html.Label(
                                            "Feature Lag (months):",
                                            style={
                                                'fontWeight': '500',
                                                'marginBottom': '8px',
                                                'display': 'block',
                                                'color': '#E0E0E0',
                                                'fontSize': '13px'
                                            }
                                        ),
                                        dcc.Input(
                                            id="lag",
                                            type="number",
                                            value=4,
                                            min=1,
                                            max=12,
                                            style={
                                                'backgroundColor': '#0a0a0a',
                                                'color': '#FF1744',
                                                'border': '1px solid rgba(255,23,68,0.4)',
                                                'borderRadius': '6px',
                                                'padding': '10px',
                                                'width': '100%',
                                                'fontSize': '13px',
                                                'fontFamily': 'JetBrains Mono, monospace',
                                                'fontWeight': '500'
                                            }
                                        ),
                                    ])
                                ]
                            )
                        ]
                    ),
                    
                    # Right Column - Model Hyperparameters
                    html.Div(
                        style={
                            'background': 'rgba(26,26,26,0.8)',
                            'border': '1px solid rgba(255,23,68,0.3)',
                            'borderRadius': '10px',
                            'padding': '25px',
                            'boxShadow': '0 4px 20px rgba(0,0,0,0.5)'
                        },
                        children=[
                            html.H3(
                                "⚙️ MODEL HYPERPARAMETERS",
                                style={
                                    'color': '#FF1744',
                                    'marginBottom': '20px',
                                    'fontSize': '16px',
                                    'fontWeight': '700',
                                    'letterSpacing': '1px',
                                    'borderBottom': '2px solid rgba(255,23,68,0.3)',
                                    'paddingBottom': '10px'
                                }
                            ),
                            
                            html.Div(
                                style={'marginBottom': '25px'},
                                children=[
                                    html.Label(
                                        "Lasso Alpha (Regularization Strength):",
                                        style={
                                            'fontWeight': '500',
                                            'marginBottom': '12px',
                                            'display': 'block',
                                            'color': '#E0E0E0',
                                            'fontSize': '13px'
                                        }
                                    ),
                                    dcc.Slider(
                                        id="alpha",
                                        min=0.01,
                                        max=0.10,
                                        step=0.01,
                                        value=0.03,
                                        marks={
                                            i/100: {
                                                'label': f'{i/100:.2f}',
                                                'style': {'color': '#FFD700', 'fontSize': '11px', 'fontWeight': '500'}
                                            }
                                            for i in range(1, 11, 2)
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                            "style": {"fontSize": "12px", "fontFamily": "JetBrains Mono"}
                                        },
                                        className='custom-slider'
                                    ),
                                ]
                            ),
                            
                            html.Div(
                                children=[
                                    html.Label(
                                        "Test Size (Train/Test Split Ratio):",
                                        style={
                                            'fontWeight': '500',
                                            'marginBottom': '12px',
                                            'display': 'block',
                                            'color': '#E0E0E0',
                                            'fontSize': '13px'
                                        }
                                    ),
                                    dcc.Slider(
                                        id="test-size",
                                        min=0.10,
                                        max=0.50,
                                        step=0.01,
                                        value=0.25,
                                        marks={
                                            i/10: {
                                                'label': f'{i/10:.1f}',
                                                'style': {'color': '#FFD700', 'fontSize': '11px', 'fontWeight': '500'}
                                            }
                                            for i in range(1, 6)
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                            "style": {"fontSize": "12px", "fontFamily": "JetBrains Mono"}
                                        },
                                        className='custom-slider'
                                    ),
                                ]
                            )
                        ]
                    ),
                ]
            ),
            
            # Run Button
            html.Div(
                style={
                    'textAlign': 'center',
                    'marginBottom': '40px'
                },
                children=[
                    html.Button(
                        "🚀 RUN MODEL EVALUATION",
                        id="button",
                        n_clicks=0,
                        style={
                            'background': 'linear-gradient(135deg, #00F5FF 0%, #FF1744 100%)',
                            'color': 'white',
                            'border': 'none',
                            'padding': '16px 50px',
                            'fontSize': '16px',
                            'fontWeight': '700',
                            'cursor': 'pointer',
                            'borderRadius': '8px',
                            'letterSpacing': '2px',
                            'textTransform': 'uppercase',
                            'boxShadow': '0 6px 30px rgba(0,245,255,0.4)',
                            'transition': 'all 0.3s ease',
                            'fontFamily': 'JetBrains Mono, monospace'
                        }
                    )
                ]
            ),
            
            # Graph Output Section
            html.Div(
                style={
                    'background': 'rgba(26,26,26,0.6)',
                    'border': '1px solid rgba(255,215,0,0.3)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'boxShadow': '0 8px 40px rgba(0,0,0,0.6)'
                },
                children=[
                    dcc.Loading(
                        id="loading",
                        type="circle",
                        color="#00F5FF",
                        children=[
                            dcc.Graph(
                                id="graph",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'fed_funds_prediction',
                                        'height': 900,
                                        'width': 1600,
                                        'scale': 2
                                    }
                                },
                                style={'backgroundColor': 'transparent'}
                            )
                        ]
                    )
                ]
            ),
            
            # Footer
            html.Div(
                style={
                    'textAlign': 'center',
                    'marginTop': '40px',
                    'paddingTop': '20px',
                    'borderTop': '1px solid rgba(255,255,255,0.1)',
                    'color': '#666',
                    'fontSize': '12px'
                },
                children=[
                    html.P("© 2026 Fed Funds Rate Predictor | Powered by FRED & Scikit-Learn")
                ]
            )
        ]
    )
    
    # Callback
    @app.callback(
        Output("graph", "figure"),
        Input("button", "n_clicks"),
        [
            State("features", "value"),
            State("date", "value"),
            State("lag", "value"),
            State("test-size", "value"),
            State("alpha", "value")
        ]
    )
    def update_graph(n_clicks, selected_features, start_date, offset, test_size, alpha):
        """Update graph based on user inputs."""
        if not n_clicks:
            # Return empty figure with dark theme
            return go.Figure().update_layout(
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='white'),
                title=dict(
                    text="<b>Click 'RUN MODEL EVALUATION' to start</b>",
                    font=dict(size=18, color='#888'),
                    x=0.5,
                    xanchor='center'
                ),
                height=900
            )
        
        # Load and prepare data
        all_features = selected_features + ["FEDFUNDS"]
        data_frames = []
        
        for feature_name in all_features:
            feature_df = pd.read_sql(
                f"SELECT * FROM {feature_name}",
                con=con,
                index_col="Datetime"
            )
            data_frames.append(feature_df)
        
        df = pd.concat(data_frames, join="inner", axis=1)
        df.index = pd.to_datetime(df.index)
        df.interpolate(inplace=True)
        
        # Filter by date and calculate returns
        df = df.loc[df.index >= pd.to_datetime(start_date)]
        df = df.pct_change().rolling(3).mean().dropna() * 100
        
        # Evaluate model and return figure
        fig = evaluate_model(df, offset, test_size, alpha)
        return fig
    
    return app


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Initialize components
    fred_client = initialize_fred_api()
    database_connection = setup_database(fred_client)
    features = get_available_features(database_connection)
    
    # Create and run app
    app = create_app(database_connection, features)
    app.run(debug=True, host='0.0.0.0', port=8050)

# ==================== Configuration ====================
class Config:
    """Application configuration and FRED feature IDs."""
    FEATURE_IDS = [
        "CPIAUCSL",   # Consumer Price Index
        "PPIACO",     # Producer Price Index
        "UNRATE",     # Unemployment Rate
        "PAYEMS",     # Total Nonfarm Payrolls
        "PCEPILFE",   # Core PCE Price Index
        "M2SL",       # M2 Money Stock
        "FEDFUNDS",   # Federal Funds Rate (Target)
        "UMCSENT"     # Consumer Sentiment
    ]
    
    FEATURE_LABELS = {
        "CPIAUCSL": "Consumer Price Index",
        "PPIACO": "Producer Price Index",
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls",
        "PCEPILFE": "Core PCE Inflation",
        "M2SL": "M2 Money Supply",
        "UMCSENT": "Consumer Sentiment"
    }
    
    DB_PATH = "data/macrodata.db"
    CONFIG_PATH = "data/configs.json"


# ==================== Data Management ====================
def initialize_fred_api() -> Fred:
    """Load API key and initialize FRED API client."""
    with open(Config.CONFIG_PATH, "r") as file:
        api_key = json.load(file)["FRED_API_KEY"]
    return Fred(api_key)


def setup_database(fred: Fred) -> sqlite3.Connection:
    """Download FRED data and populate SQLite database."""
    con = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
    
    for feature_id in Config.FEATURE_IDS:
        series = fred.get_series(feature_id).rename(feature_id)
        series.to_sql(
            name=feature_id,
            con=con,
            if_exists="replace",
            index_label="Datetime"
        )
    
    return con


def get_available_features(con: sqlite3.Connection) -> List[str]:
    """Retrieve list of available feature tables (excluding target variable)."""
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [name[0] for name in cur.fetchall()]
    
    if "FEDFUNDS" in tables:
        tables.remove("FEDFUNDS")
    
    return tables


# ==================== Model Evaluation ====================
def evaluate_model(
    df: pd.DataFrame,
    offset: int,
    test_size: float,
    alpha: float
) -> go.Figure:
    """
    Train Lasso model and create comprehensive evaluation plots.
    
    Args:
        df: DataFrame with features and target
        offset: Number of periods to lag features
        test_size: Proportion of data for testing
        alpha: Lasso regularization parameter
        
    Returns:
        Plotly figure with 4 evaluation subplots
    """
    df = df.dropna()
    
    # Prepare lagged features
    shifted_features = df.drop("FEDFUNDS", axis=1).shift(int(offset)).dropna()
    valid_idx = shifted_features.index
    
    # Scale features and split data
    scaler = StandardScaler()
    X = scaler.fit_transform(shifted_features)
    y = np.array(df.FEDFUNDS.loc[valid_idx])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=25, test_size=float(test_size)
    )
    
    # Train model
    model = Lasso(
        alpha=float(alpha),
        fit_intercept=True,
        max_iter=1200,
        warm_start=True,
        random_state=25
    )
    model.fit(X=X_train, y=y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test).flatten()
    y_test = y_test.flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(model, X_train, y_train, y_test, y_pred)
    
    # Create visualization
    fig = create_evaluation_plots(
        y_test, y_pred, shifted_features.columns, model.coef_, metrics
    )
    
    return fig


def calculate_metrics(
    model: Lasso,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate model performance metrics."""
    cv_scores = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=TimeSeriesSplit(n_splits=10),
        scoring='r2'
    )
    
    metrics = {
        "R²": r2_score(y_true=y_test, y_pred=y_pred),
        "R² CV": np.median(cv_scores),
        "RMSE": np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)),
        "MAE": mean_absolute_error(y_true=y_test, y_pred=y_pred)
    }
    
    # Print metrics to console
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    for key, val in metrics.items():
        print(f"{key:20s}: {val:.4f}")
    print("="*60 + "\n")
    
    return metrics


def create_evaluation_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    coefficients: np.ndarray,
    metrics: Dict[str, float]
) -> go.Figure:
    """Create professional evaluation plots with premium dark theme."""
    
    # Create subplot structure with proper spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Distribution Comparison (KDE)</b>",
            "<b>Time Series: Actual vs Predicted</b>",
            "<b>Prediction Scatter Plot</b>",
            "<b>Feature Importance (|Coefficients|)</b>"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.14,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # Premium color scheme - Neon on Dark
    COLOR_ACTUAL = "#00F5FF"      # Electric Cyan
    COLOR_PRED = "#FF1744"        # Neon Red
    COLOR_ACCENT = "#FFD700"      # Gold
    COLOR_GRID = "#1a1a1a"
    COLOR_TEXT = "#E0E0E0"
    
    # 1. KDE Distribution Plot
    y_test_kde = gaussian_kde(y_test)
    y_pred_kde = gaussian_kde(y_pred)
    mean, std = np.mean(y_test), np.std(y_test)
    x_range = np.linspace(mean - 3*std, mean + 3*std, 1000)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_test_kde(x_range),
            mode="lines",
            name="Actual Distribution",
            line=dict(color=COLOR_ACTUAL, width=3),
            fill='tozeroy',
            fillcolor=f'rgba(0, 245, 255, 0.15)',
            hovertemplate='<b>Rate:</b> %{x:.2f}%<br><b>Density:</b> %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred_kde(x_range),
            mode="lines",
            name="Predicted Distribution",
            line=dict(color=COLOR_PRED, width=3, dash='dot'),
            fill='tozeroy',
            fillcolor=f'rgba(255, 23, 68, 0.15)',
            hovertemplate='<b>Rate:</b> %{x:.2f}%<br><b>Density:</b> %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Time Series Plot
    x_axis = np.arange(len(y_test))
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_test,
            name="Actual Values",
            mode="lines+markers",
            line=dict(color=COLOR_ACTUAL, width=2.5),
            marker=dict(size=5, symbol='circle', line=dict(width=1, color='white')),
            hovertemplate='<b>Index:</b> %{x}<br><b>Actual Rate:</b> %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_pred,
            name="Predictions",
            mode="lines+markers",
            line=dict(color=COLOR_PRED, width=2.5, dash='dash'),
            marker=dict(size=5, symbol='diamond', line=dict(width=1, color='white')),
            hovertemplate='<b>Index:</b> %{x}<br><b>Predicted Rate:</b> %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Scatter Plot with perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=y_test,
            mode="markers",
            name="Predictions",
            marker=dict(
                size=10,
                color=y_test,
                colorscale='Plasma',
                showscale=True,
                line=dict(width=1.5, color='rgba(255,255,255,0.6)'),
                colorbar=dict(
                    title=dict(text="<b>Actual<br>Rate (%)</b>", font=dict(size=11)),
                    x=0.46,
                    len=0.35,
                    thickness=12,
                    tickfont=dict(size=10, color=COLOR_TEXT),
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='rgba(255,255,255,0.3)',
                    borderwidth=1
                )
            ),
            showlegend=False,
            hovertemplate='<b>Predicted:</b> %{x:.2f}%<br><b>Actual:</b> %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min()) - 0.5
    max_val = max(y_test.max(), y_pred.max()) + 0.5
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color=COLOR_ACCENT, width=2, dash="dot"),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # 4. Feature Importance Bar Chart
    abs_coefs = np.abs(coefficients)
    colors_bars = [COLOR_ACCENT if c == abs_coefs.max() else COLOR_PRED for c in abs_coefs]
    
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=abs_coefs,
            name="Coefficient Magnitude",
            marker=dict(
                color=colors_bars,
                line=dict(color='rgba(255,255,255,0.4)', width=1.5),
                pattern=dict(shape="/", bgcolor="rgba(0,0,0,0.3)", size=4, solidity=0.3)
            ),
            showlegend=False,
            hovertemplate='<b>%{x}</b><br><b>|Coefficient|:</b> %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update all axes with consistent styling
    axis_style = dict(
        showgrid=True,
        gridcolor=COLOR_GRID,
        gridwidth=1,
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.2)',
        zerolinewidth=2,
        color=COLOR_TEXT,
        tickfont=dict(size=10, family='JetBrains Mono, Consolas, monospace'),
        title=dict(font=dict(size=12, family='JetBrains Mono, Consolas, monospace', color='white'))
    )
    
    # Apply axis labels and styling
    fig.update_xaxes(title_text="<b>Fed Funds Rate (%)</b>", row=1, col=1, **axis_style)
    fig.update_yaxes(title_text="<b>Probability Density</b>", row=1, col=1, **axis_style)
    
    fig.update_xaxes(title_text="<b>Test Sample Index</b>", row=1, col=2, **axis_style)
    fig.update_yaxes(title_text="<b>Fed Funds Rate (%)</b>", row=1, col=2, **axis_style)
    
    fig.update_xaxes(title_text="<b>Predicted Rate (%)</b>", row=2, col=1, **axis_style)
    fig.update_yaxes(title_text="<b>Actual Rate (%)</b>", row=2, col=1, **axis_style)
    
    fig.update_xaxes(title_text="<b>Economic Indicator</b>", row=2, col=2, **axis_style, tickangle=-35)
    fig.update_yaxes(title_text="<b>|Coefficient|</b>", row=2, col=2, **axis_style)
    
    # Overall layout with premium dark theme
    fig.update_layout(
        height=900,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(
            color='white',
            size=11,
            family='JetBrains Mono, Consolas, monospace'
        ),
        title=dict(
            text=(
                f"<b style='font-size:24px; color:#00F5FF;'>FED FUNDS RATE PREDICTION MODEL</b><br>"
                f"<span style='font-size:14px; color:#888;'>Lasso Regression with Macroeconomic Indicators</span><br>"
                f"<span style='font-size:13px; color:#FFD700;'>"
                f"R² = {metrics['R²']:.4f} │ "
                f"R² CV = {metrics['R² CV']:.4f} │ "
                f"RMSE = {metrics['RMSE']:.4f} │ "
                f"MAE = {metrics['MAE']:.4f}"
                f"</span>"
            ),
            font=dict(family='JetBrains Mono, Consolas, monospace'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26,26,26,0.9)',
            bordercolor='rgba(0, 245, 255, 0.5)',
            borderwidth=2,
            font=dict(size=11, family='JetBrains Mono, Consolas, monospace'),
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top'
        ),
        margin=dict(t=160, b=80, l=80, r=80),
        hovermode='closest'
    )
    
    # Update subplot titles styling
    for annotation in fig['layout']['annotations'][:4]:
        annotation['font'] = dict(
            size=13,
            color='#00F5FF',
            family='JetBrains Mono, Consolas, monospace'
        )
        annotation['y'] = annotation['y'] + 0.01
    
    return fig


# ==================== Dash Application ====================
def create_app(con: sqlite3.Connection, available_features: List[str]) -> Dash:
    """Create and configure Dash application with premium dark theme."""
    
    app = Dash(__name__)
    
    # External CSS for premium fonts
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Fed Funds Rate Predictor</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    background: linear-gradient(135deg, #000000 0%, #0a0a0a 50%, #000000 100%);
                    background-attachment: fixed;
                    min-height: 100vh;
                }
                
                /* Animated gradient background effect */
                @keyframes gradient {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                
                /* Custom scrollbar */
                ::-webkit-scrollbar {
                    width: 10px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #0a0a0a;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: linear-gradient(180deg, #00F5FF, #FF1744);
                    border-radius: 5px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(180deg, #00D9FF, #FF0033);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    app.layout = html.Div(
        style={
            'backgroundColor': 'transparent',
            'color': 'white',
            'fontFamily': 'JetBrains Mono, Consolas, monospace',
            'padding': '40px 60px',
            'maxWidth': '1600px',
            'margin': '0 auto'
        },
        children=[
            # Header Section with Gradient Border
            html.Div(
                style={
                    'background': 'linear-gradient(135deg, rgba(0,245,255,0.1) 0%, rgba(255,23,68,0.1) 100%)',
                    'border': '2px solid',
                    'borderImage': 'linear-gradient(135deg, #00F5FF, #FF1744) 1',
                    'borderRadius': '12px',
                    'padding': '30px',
                    'marginBottom': '40px',
                    'boxShadow': '0 0 40px rgba(0,245,255,0.2)',
                    'animation': 'fadeIn 0.8s ease-out'
                },
                children=[
                    html.H1(
                        "⚡ FED FUNDS RATE PREDICTOR",
                        style={
                            'textAlign': 'center',
                            'marginBottom': '10px',
                            'background': 'linear-gradient(135deg, #00F5FF 0%, #FFD700 50%, #FF1744 100%)',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent',
                            'fontSize': '42px',
                            'fontWeight': '700',
                            'letterSpacing': '2px',
                            'textTransform': 'uppercase'
                        }
                    ),
                    html.P(
                        "Advanced Lasso Regression with Macroeconomic Time Series Analysis",
                        style={
                            'textAlign': 'center',
                            'color': '#888',
                            'fontSize': '14px',
                            'letterSpacing': '1px',
                            'fontWeight': '400'
                        }
                    ),
                ]
            ),
            
            # Control Panel Section
            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '30px',
                    'marginBottom': '40px'
                },
                children=[
                    # Left Column - Feature Selection & Date
                    html.Div(
                        style={
                            'background': 'rgba(26,26,26,0.8)',
                            'border': '1px solid rgba(0,245,255,0.3)',
                            'borderRadius': '10px',
                            'padding': '25px',
                            'boxShadow': '0 4px 20px rgba(0,0,0,0.5)'
                        },
                        children=[
                            html.H3(
                                "📊 DATA CONFIGURATION",
                                style={
                                    'color': '#00F5FF',
                                    'marginBottom': '20px',
                                    'fontSize': '16px',
                                    'fontWeight': '700',
                                    'letterSpacing': '1px',
                                    'borderBottom': '2px solid rgba(0,245,255,0.3)',
                                    'paddingBottom': '10px'
                                }
                            ),
                            
                            html.Label(
                                "Select Economic Indicators:",
                                style={
                                    'fontWeight': '500',
                                    'marginBottom': '8px',
                                    'display': 'block',
                                    'color': '#E0E0E0',
                                    'fontSize': '13px'
                                }
                            ),
                            dcc.Dropdown(
                                id="features",
                                options=[
                                    {'label': f"{feat} - {Config.FEATURE_LABELS.get(feat, feat)}", 
                                     'value': feat} 
                                    for feat in available_features
                                ],
                                value=available_features[:3],
                                multi=True,
                                style={
                                    'marginBottom': '20px',
                                },
                                className='custom-dropdown'
                            ),
                            
                            html.Div(
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': '1fr 1fr',
                                    'gap': '15px'
                                },
                                children=[
                                    html.Div([
                                        html.Label(
                                            "Start Date:",
                                            style={
                                                'fontWeight': '500',
                                                'marginBottom': '8px',
                                                'display': 'block',
                                                'color': '#E0E0E0',
                                                'fontSize': '13px'
                                            }
                                        ),
                                        dcc.Input(
                                            id="date",
                                            type="text",
                                            value="2019-01-01",
                                            placeholder="YYYY-MM-DD",
                                            style={
                                                'backgroundColor': '#0a0a0a',
                                                'color': '#00F5FF',
                                                'border': '1px solid rgba(0,245,255,0.4)',
                                                'borderRadius': '6px',
                                                'padding': '10px',
                                                'width': '100%',
                                                'fontSize': '13px',
                                                'fontFamily': 'JetBrains Mono, monospace',
                                                'fontWeight': '500'
                                            }
                                        ),
                                    ]),
                                    
                                    html.Div([
                                        html.Label(
                                            "Feature Lag (months):",
                                            style={
                                                'fontWeight': '500',
                                                'marginBottom': '8px',
                                                'display': 'block',
                                                'color': '#E0E0E0',
                                                'fontSize': '13px'
                                            }
                                        ),
                                        dcc.Input(
                                            id="lag",
                                            type="number",
                                            value=4,
                                            min=1,
                                            max=12,
                                            style={
                                                'backgroundColor': '#0a0a0a',
                                                'color': '#FF1744',
                                                'border': '1px solid rgba(255,23,68,0.4)',
                                                'borderRadius': '6px',
                                                'padding': '10px',
                                                'width': '100%',
                                                'fontSize': '13px',
                                                'fontFamily': 'JetBrains Mono, monospace',
                                                'fontWeight': '500'
                                            }
                                        ),
                                    ])
                                ]
                            )
                        ]
                    ),
                    
                    # Right Column - Model Hyperparameters
                    html.Div(
                        style={
                            'background': 'rgba(26,26,26,0.8)',
                            'border': '1px solid rgba(255,23,68,0.3)',
                            'borderRadius': '10px',
                            'padding': '25px',
                            'boxShadow': '0 4px 20px rgba(0,0,0,0.5)'
                        },
                        children=[
                            html.H3(
                                "⚙️ MODEL HYPERPARAMETERS",
                                style={
                                    'color': '#FF1744',
                                    'marginBottom': '20px',
                                    'fontSize': '16px',
                                    'fontWeight': '700',
                                    'letterSpacing': '1px',
                                    'borderBottom': '2px solid rgba(255,23,68,0.3)',
                                    'paddingBottom': '10px'
                                }
                            ),
                            
                            html.Div(
                                style={'marginBottom': '25px'},
                                children=[
                                    html.Label(
                                        "Lasso Alpha (Regularization Strength):",
                                        style={
                                            'fontWeight': '500',
                                            'marginBottom': '12px',
                                            'display': 'block',
                                            'color': '#E0E0E0',
                                            'fontSize': '13px'
                                        }
                                    ),
                                    dcc.Slider(
                                        id="alpha",
                                        min=0.01,
                                        max=0.10,
                                        step=0.01,
                                        value=0.03,
                                        marks={
                                            i/100: {
                                                'label': f'{i/100:.2f}',
                                                'style': {'color': '#FFD700', 'fontSize': '11px', 'fontWeight': '500'}
                                            }
                                            for i in range(1, 11, 2)
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                            "style": {"fontSize": "12px", "fontFamily": "JetBrains Mono"}
                                        },
                                        className='custom-slider'
                                    ),
                                ]
                            ),
                            
                            html.Div(
                                children=[
                                    html.Label(
                                        "Test Size (Train/Test Split Ratio):",
                                        style={
                                            'fontWeight': '500',
                                            'marginBottom': '12px',
                                            'display': 'block',
                                            'color': '#E0E0E0',
                                            'fontSize': '13px'
                                        }
                                    ),
                                    dcc.Slider(
                                        id="test-size",
                                        min=0.10,
                                        max=0.50,
                                        step=0.01,
                                        value=0.25,
                                        marks={
                                            i/10: {
                                                'label': f'{i/10:.1f}',
                                                'style': {'color': '#FFD700', 'fontSize': '11px', 'fontWeight': '500'}
                                            }
                                            for i in range(1, 6)
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                            "style": {"fontSize": "12px", "fontFamily": "JetBrains Mono"}
                                        },
                                        className='custom-slider'
                                    ),
                                ]
                            )
                        ]
                    ),
                ]
            ),
            
            # Run Button
            html.Div(
                style={
                    'textAlign': 'center',
                    'marginBottom': '40px'
                },
                children=[
                    html.Button(
                        "🚀 RUN MODEL EVALUATION",
                        id="button",
                        n_clicks=0,
                        style={
                            'background': 'linear-gradient(135deg, #00F5FF 0%, #FF1744 100%)',
                            'color': 'white',
                            'border': 'none',
                            'padding': '16px 50px',
                            'fontSize': '16px',
                            'fontWeight': '700',
                            'cursor': 'pointer',
                            'borderRadius': '8px',
                            'letterSpacing': '2px',
                            'textTransform': 'uppercase',
                            'boxShadow': '0 6px 30px rgba(0,245,255,0.4)',
                            'transition': 'all 0.3s ease',
                            'fontFamily': 'JetBrains Mono, monospace'
                        }
                    )
                ]
            ),
            
            # Graph Output Section
            html.Div(
                style={
                    'background': 'rgba(26,26,26,0.6)',
                    'border': '1px solid rgba(255,215,0,0.3)',
                    'borderRadius': '12px',
                    'padding': '20px',
                    'boxShadow': '0 8px 40px rgba(0,0,0,0.6)'
                },
                children=[
                    dcc.Loading(
                        id="loading",
                        type="circle",
                        color="#00F5FF",
                        children=[
                            dcc.Graph(
                                id="graph",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'fed_funds_prediction',
                                        'height': 900,
                                        'width': 1600,
                                        'scale': 2
                                    }
                                },
                                style={'backgroundColor': 'transparent'}
                            )
                        ]
                    )
                ]
            ),
            
            # Footer
            html.Div(
                style={
                    'textAlign': 'center',
                    'marginTop': '40px',
                    'paddingTop': '20px',
                    'borderTop': '1px solid rgba(255,255,255,0.1)',
                    'color': '#666',
                    'fontSize': '12px'
                },
                children=[
                    html.P("© 2026 Fed Funds Rate Predictor | Powered by FRED & Scikit-Learn")
                ]
            )
        ]
    )
    
    # Callback
    @app.callback(
        Output("graph", "figure"),
        Input("button", "n_clicks"),
        [
            State("features", "value"),
            State("date", "value"),
            State("lag", "value"),
            State("test-size", "value"),
            State("alpha", "value")
        ]
    )
    def update_graph(n_clicks, selected_features, start_date, offset, test_size, alpha):
        """Update graph based on user inputs."""
        if not n_clicks:
            # Return empty figure with dark theme
            return go.Figure().update_layout(
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='white'),
                title=dict(
                    text="<b>Click 'RUN MODEL EVALUATION' to start</b>",
                    font=dict(size=18, color='#888'),
                    x=0.5,
                    xanchor='center'
                ),
                height=900
            )
        
        # Load and prepare data
        all_features = selected_features + ["FEDFUNDS"]
        data_frames = []
        
        for feature_name in all_features:
            feature_df = pd.read_sql(
                f"SELECT * FROM {feature_name}",
                con=con,
                index_col="Datetime"
            )
            data_frames.append(feature_df)
        
        df = pd.concat(data_frames, join="inner", axis=1)
        df.index = pd.to_datetime(df.index)
        df.interpolate(inplace=True)
        
        # Filter by date and calculate returns
        df = df.loc[df.index >= pd.to_datetime(start_date)]
        df = df.pct_change().rolling(3).mean().dropna() * 100
        
        # Evaluate model and return figure
        fig = evaluate_model(df, offset, test_size, alpha)
        return fig
    
    return app


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Initialize components
    fred_client = initialize_fred_api()
    database_connection = setup_database(fred_client)
    features = get_available_features(database_connection)
    
    # Create and run app
    app = create_app(database_connection, features)
    app.run()