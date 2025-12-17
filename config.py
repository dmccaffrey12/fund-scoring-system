# config.py
"""
Configuration file for Fund Scoring System 2025
Contains all weights, thresholds, and color schemes
"""

# ============================================================================
# PASSIVE FUND WEIGHTS (100%)
# ============================================================================
PASSIVE_WEIGHTS = {
    'expense_ratio': 0.40,
    'tracking_error_3y': 0.11,
    'tracking_error_5y': 0.11,
    'tracking_error_10y': 0.11,
    'r_squared_5y': 0.05,
    'aum': 0.06,
    'downside_capture_5y': 0.06,
    'downside_capture_10y': 0.04,
    'max_drawdown_5y': 0.04,
    'max_drawdown_10y': 0.02,
}

# ============================================================================
# ACTIVE FUND WEIGHTS (100%)
# ============================================================================
ACTIVE_WEIGHTS = {
    'expense_ratio': 0.25,
    'information_ratio_3y': 0.10,
    'information_ratio_5y': 0.06,
    'information_ratio_10y': 0.04,
    'sortino_ratio_3y': 0.10,
    'sortino_ratio_5y': 0.06,
    'sortino_ratio_10y': 0.04,
    'max_drawdown_5y': 0.07,
    'max_drawdown_10y': 0.03,
    'downside_capture_5y': 0.07,
    'downside_capture_10y': 0.03,
    'returns_3y': 0.01,
    'returns_5y': 0.01,
    'returns_10y': 0.02,
    'upside_capture_5y': 0.06,
    'upside_capture_10y': 0.05,
}

# ============================================================================
# CSV COLUMN MAPPINGS
# ============================================================================
CSV_COLUMNS = {
    'symbol': 'Symbol',
    'name': 'Name',
    'index_fund': 'Index Fund',
    'category': 'Category Name',
    'expense_ratio': 'Net Expense Ratio',
    'tracking_error_3y': 'Tracking Error (vs Category) (3Y)',
    'tracking_error_5y': 'Tracking Error (vs Category) (5Y)',
    'tracking_error_10y': 'Tracking Error (vs Category) (10Y)',
    'r_squared_5y': 'R-Squared (vs Category) (5Y)',
    'aum': 'Share Class Assets Under Management',
    'downside_capture_5y': 'Downside (vs Category) (5Y)',
    'downside_capture_10y': 'Downside (vs Category) (10Y)',
    'max_drawdown_5y': 'Max Drawdown (5Y)',
    'max_drawdown_10y': 'Max Drawdown (10Y)',
    'information_ratio_3y': 'Information Ratio (vs Category) (3Y)',
    'information_ratio_5y': 'Information Ratio (vs Category) (5Y)',
    'information_ratio_10y': 'Information Ratio (vs Category) (10Y)',
    'sortino_ratio_3y': 'Historical Sortino (3Y)',
    'sortino_ratio_5y': 'Historical Sortino (5Y)',
    'sortino_ratio_10y': 'Historical Sortino (10Y)',
    'upside_capture_3y': 'Upside (vs Category) (3Y)',
    'upside_capture_5y': 'Upside (vs Category) (5Y)',
    'upside_capture_10y': 'Upside (vs Category) (10Y)',
    'returns_3y': '3 Year Total Returns (Daily)',
    'returns_5y': '5 Year Total Returns (Daily)',
    'returns_10y': '10 Year Total Returns (Daily)',
}

# ============================================================================
# SCORE COLOR SCHEME
# ============================================================================
SCORE_COLORS = {
    'excellent': '#00AA00',  # Green (80-100)
    'good': '#FFFF00',       # Yellow (60-79)
    'fair': '#FF9900',       # Orange (40-59)
    'poor': '#FF0000',       # Red (0-39)
}

SCORE_CATEGORIES = {
    'excellent': (80, 100),
    'good': (60, 79),
    'fair': (40, 59),
    'poor': (0, 39),
}

# ============================================================================
# THRESHOLDS
# ============================================================================
MISSING_DATA_THRESHOLD = 0.30  # Flag if >30% data missing
MIN_FUNDS_IN_CATEGORY = 5      # Minimum funds to show category
PERCENTILE_PRECISION = 2       # Decimal places for percentiles

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
DASHBOARD_REFRESH_RATE = 3600  # Seconds (cache data for 1 hour)
MAX_FUNDS_COMPARISON = 5       # Maximum funds to compare at once
DEFAULT_CHART_HEIGHT = 500     # Height in pixels for charts
DEFAULT_CHART_WIDTH = 800      # Width in pixels for charts

# ============================================================================
# METRIC CATEGORIES FOR SCORING BREAKDOWN
# ============================================================================
PASSIVE_CATEGORIES = {
    'Cost': ['expense_ratio'],
    'Tracking Accuracy': ['tracking_error_3y', 'tracking_error_5y', 'tracking_error_10y'],
    'Structural Fidelity': ['r_squared_5y'],
    'Stability': ['aum'],
    'Risk Management': ['downside_capture_5y', 'downside_capture_10y', 'max_drawdown_5y', 'max_drawdown_10y'],
}

ACTIVE_CATEGORIES = {
    'Cost': ['expense_ratio'],
    'Skill': ['information_ratio_3y', 'information_ratio_5y', 'information_ratio_10y', 
              'sortino_ratio_3y', 'sortino_ratio_5y', 'sortino_ratio_10y'],
    'Risk Management': ['max_drawdown_5y', 'max_drawdown_10y', 'downside_capture_5y', 'downside_capture_10y'],
    'Performance': ['returns_3y', 'returns_5y', 'returns_10y', 'upside_capture_5y', 'upside_capture_10y'],
}

# ============================================================================
# METRIC DIRECTION (for percentile calculation)
# Lower values better: -1, Higher values better: 1
# ============================================================================
METRIC_DIRECTION = {
    'expense_ratio': -1,           # Lower is better
    'tracking_error_3y': -1,       # Lower is better
    'tracking_error_5y': -1,       # Lower is better
    'tracking_error_10y': -1,      # Lower is better
    'r_squared_5y': 1,             # Higher is better
    'aum': 1,                       # Higher is better
    'downside_capture_5y': -1,     # Lower is better
    'downside_capture_10y': -1,    # Lower is better
    'max_drawdown_5y': -1,         # Lower is better
    'max_drawdown_10y': -1,        # Lower is better
    'information_ratio_3y': 1,     # Higher is better
    'information_ratio_5y': 1,     # Higher is better
    'information_ratio_10y': 1,    # Higher is better
    'sortino_ratio_3y': 1,         # Higher is better
    'sortino_ratio_5y': 1,         # Higher is better
    'sortino_ratio_10y': 1,        # Higher is better
    'returns_3y': 1,               # Higher is better
    'returns_5y': 1,               # Higher is better
    'returns_10y': 1,              # Higher is better
    'upside_capture_3y': 1,        # Higher is better
    'upside_capture_5y': 1,        # Higher is better
    'upside_capture_10y': 1,       # Higher is better
}
