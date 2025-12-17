import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
from datetime import datetime
import glob
import os
import json
from pathlib import Path

st.set_page_config(page_title="Fund Scoring System 2025", layout="wide")

VERSIONS_DIR = "versions"
METADATA_FILE = os.path.join(VERSIONS_DIR, "metadata.json")
os.makedirs(VERSIONS_DIR, exist_ok=True)

@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_csv():
    csvfiles = glob.glob("*.csv")
    if csvfiles:
        try:
            df = pd.read_csv(csvfiles[0])
            return df, csvfiles[0]
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None, None
    return None, None

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def save_version(df, notes):
    metadata = load_metadata()
    version_num = len(metadata) + 1
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"v{version_num}_{date_str}_fund_screener.csv"
    filepath = os.path.join(VERSIONS_DIR, filename)
    
    df.to_csv(filepath, index=False)
    
    metadata[f"v{version_num}"] = {
        "filename": filename,
        "date": date_str,
        "timestamp": datetime.now().isoformat(),
        "totalFunds": len(df),
        "avgScore": float(df.get("2025 Scoring System", pd.Series([0])).mean()) if "2025 Scoring System" in df.columns else 0,
        "notes": notes
    }
    save_metadata(metadata)
    st.success(f"Saved as Version {version_num}")
    st.cache_data.clear()

df, csvname = load_csv()

if df is None:
    st.title("Fund Scoring System 2025")
    st.markdown("---")
    st.write("No CSV file found in directory. ")
    st.write(f"Current directory: {os.getcwd()}")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} funds")
            notes = st.text_input("Add notes for this version (optional)", "")
            if st.button("Save as New Version"):
                save_version(df, notes)
                st.info("Version saved! Refresh to see it in the app")
                st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ===== CRITICAL FIX: SCORE FUNCTIONS =====
# These functions ONLY read from CSV, never calculate
@st.cache_data
def get_scores(df):
    """Load scores from CSV 2025 Scoring System column - NO CALCULATIONS"""
    if "2025 Scoring System" not in df.columns:
        st.error("2025 Scoring System column not found in CSV")
        st.write("Available columns:", list(df.columns))
        st.stop()
    # ONLY return the CSV column, nothing else
    return df[["Symbol", "2025 Scoring System"]].copy()

scores_df = get_scores(df)

def get_fund_score(symbol):
    """Get score from CSV for a specific fund - DIRECT READ ONLY, NO CALCULATIONS"""
    score_row = scores_df[scores_df["Symbol"] == symbol]
    if not score_row.empty:
        return float(score_row.iloc[0]["2025 Scoring System"])
    return None

def get_score_color(score):
    """Display color based on CSV score value"""
    if score is None:
        return "Gray"
    elif score >= 0.70:
        return "Green"
    elif score >= 0.50:
        return "Yellow"
    elif score >= 0.30:
        return "Orange"
    else:
        return "Red"

def get_score_category(score):
    """Display category based on CSV score value"""
    if score is None:
        return "N/A"
    elif score >= 0.70:
        return "Excellent"
    elif score >= 0.50:
        return "Good"
    elif score >= 0.30:
        return "Fair"
    else:
        return "Poor"

# ===== RADAR CHART: VISUALIZATION ONLY (Does NOT affect score) =====
def create_radar_chart(fund_data, fund_symbol):
    """
    Creates radar chart for VISUALIZATION ONLY.
    Uses CSV score but displays fund metrics for reference.
    DOES NOT calculate or modify the score.
    """
    score = get_fund_score(fund_symbol)
    if score is None:
        return None
    
    # These metrics are for VISUALIZATION ONLY - not for scoring
    metrics = ["Expense", "Return", "Risk", "Volatility", "Market", "Diversification"]
    
    # Calculate visualization metrics from fund data
    expense_efficiency = min(100, max(0, 0.02 - fund_data.get("Net Expense Ratio", 0.01)) / 0.02 * 100)
    return_performance = min(100, max(0, fund_data.get("5 Year Total Returns (Daily)", 0)) / 0.20 * 60 * 100)
    risk_management = min(100, max(0, 1 - fund_data.get("Max Drawdown 5Y", 0.5)) / 1 * 100)
    volatility = min(100, max(0, 1 - fund_data.get("Tracking Error vs Category 5Y", 0.1)) / 1 * 100)
    alignment = min(100, max(0, fund_data.get("R-Squared vs Category 5Y", 0.9)) / 1 * 100)
    diversification = 50  # Static for visualization
    
    values = [expense_efficiency, return_performance, risk_management, volatility, alignment, diversification, expense_efficiency]
    metrics_closed = metrics + [metrics[0]]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=metrics_closed,
        fill="toself",
        name=fund_symbol,
        line=dict(color="1E80A0"),
        fillcolor="rgba(30, 128, 160, 0.3)"
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]),),
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", 
    ["Single Fund Lookup", "Multi-Fund Comparison", "Peer Rankings", 
     "Category Overview", "Upload New Version", "Version History"])

# Pages implementation continues as in your current code...
# The KEY FIX above is the get_fund_score() function that reads CSV directly

if page == "Single Fund Lookup":
    st.title("Single Fund Lookup")
    st.markdown("---")
    
    search_symbol = st.text_input("Enter fund symbol", placeholder="e.g., AVLC, PRILX").upper()
    
    if search_symbol:
        fund = df[df["Symbol"] == search_symbol]
        
        if not fund.empty:
            fund = fund.iloc[0]
            score = get_fund_score(search_symbol)  # READS FROM CSV DIRECTLY
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Symbol", search_symbol)
            with col2:
                st.metric("Name", fund.get("Name", "N/A")[:30])
            with col3:
                st.metric("Category", fund.get("Category Name", "N/A")[:20])
            with col4:
                st.metric(f"{get_score_color(score)} Score", f"{score:.4f}")
        else:
            st.error(f"Fund {search_symbol} not found")

# Rest of the pages implementation
st.markdown("---")
st.write(f"Data Source: {csvname} | Total Funds: {len(df)} | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")