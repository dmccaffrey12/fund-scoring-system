# app.py
"""
Fund Scoring System 2025 - Streamlit Dashboard
Multi-page application for scoring 4,000+ mutual funds and ETFs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from config import (
    PASSIVE_WEIGHTS, ACTIVE_WEIGHTS, SCORE_COLORS, SCORE_CATEGORIES,
    CSV_COLUMNS, MISSING_DATA_THRESHOLD, PASSIVE_CATEGORIES, ACTIVE_CATEGORIES
)
from scoring_engine import FundScoringEngine

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fund Scoring System 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CACHE & SESSION STATE
# ============================================================================

@st.cache_data
def load_data():
    """Load and prepare fund data"""
    df = pd.read_csv('fund_screener_data-41.csv')
    return df

@st.cache_data
def compute_scores(_df):
    """Compute all fund scores"""
    engine = FundScoringEngine(_df)
    scores = engine.calculate_all_scores()
    return scores, engine

def get_score_category(score):
    """Get color and category name for a score"""
    if pd.isna(score):
        return 'unknown', 'N/A'
    if score >= 80:
        return 'excellent', 'Excellent'
    elif score >= 60:
        return 'good', 'Good'
    elif score >= 40:
        return 'fair', 'Fair'
    else:
        return 'poor', 'Poor'

def format_currency(value):
    """Format large numbers as currency"""
    if pd.isna(value):
        return 'N/A'
    if value >= 1e9:
        return f'${value/1e9:.2f}B'
    elif value >= 1e6:
        return f'${value/1e6:.2f}M'
    else:
        return f'${value:,.0f}'

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Load data
    df = load_data()
    scores, engine = compute_scores(df)
    
    # Merge scores with fund data
    df_with_scores = df.merge(scores, on='Symbol', how='left')
    
    # Sidebar navigation
    st.sidebar.title("📈 Fund Scoring System 2025")
    st.sidebar.write("Professional fund evaluation dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Dashboard Page",
        ["🔍 Single Fund Lookup", "⚖️ Multi-Fund Comparison", 
         "🏆 Peer Rankings", "📊 Category Overview"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Total Funds:** {len(df):,}")
    st.sidebar.write(f"**Passive:** {len(df[df['Index Fund']==True]):,}")
    st.sidebar.write(f"**Active:** {len(df[df['Index Fund']==False]):,}")
    
    # Route to pages
    if page == "🔍 Single Fund Lookup":
        page_single_fund(df_with_scores)
    elif page == "⚖️ Multi-Fund Comparison":
        page_multi_comparison(df_with_scores)
    elif page == "🏆 Peer Rankings":
        page_peer_rankings(df_with_scores)
    else:
        page_category_overview(df_with_scores)

# ============================================================================
# PAGE 1: SINGLE FUND LOOKUP
# ============================================================================

def page_single_fund(df):
    """Single fund lookup page"""
    st.title("🔍 Single Fund Lookup")
    st.write("Search for a fund and view detailed scoring analysis")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by ticker or fund name",
            placeholder="e.g., VOO, VTSAX, Apple"
        )
    
    with col2:
        fund_type_filter = st.selectbox(
            "Fund Type",
            ["All", "Passive", "Active"]
        )
    
    # Filter data
    filtered_df = df.copy()
    
    if fund_type_filter == "Passive":
        filtered_df = filtered_df[filtered_df['Index Fund'] == True]
    elif fund_type_filter == "Active":
        filtered_df = filtered_df[filtered_df['Index Fund'] == False]
    
    if search_query:
        mask = (
            filtered_df['Symbol'].str.contains(search_query.upper(), na=False) |
            filtered_df['Name'].str.contains(search_query, na=False, case=False)
        )
        filtered_df = filtered_df[mask]
    
    # Display results
    if len(filtered_df) == 0:
        st.warning("No funds found. Try a different search.")
        return
    
    # Select fund
    selected_symbol = st.selectbox(
        "Select Fund",
        filtered_df['Symbol'].values,
        format_func=lambda x: f"{x} - {filtered_df[filtered_df['Symbol']==x]['Name'].values[0]}"
    )
    
    fund_data = filtered_df[filtered_df['Symbol'] == selected_symbol].iloc[0]
    
    st.markdown("---")
    
    # Display fund header
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ticker", fund_data['Symbol'])
    with col2:
        fund_type = "Passive" if fund_data['Index Fund'] else "Active"
        st.metric("Fund Type", fund_type)
    with col3:
        category = fund_data['Category Name']
        st.metric("Category", category)
    
    st.markdown("---")
    
    # Score display
    score = fund_data['Score']
    category_key, category_name = get_score_category(score)
    color = SCORE_COLORS.get(category_key, '#888888')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if pd.notna(score):
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: {color}; border-radius: 10px; color: white;'>
                <h1>{score:.1f}</h1>
                <p>Score</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Score not available (insufficient data)")
    
    with col2:
        percentile = fund_data['Percentile']
        if pd.notna(percentile):
            st.metric("Percentile Rank", f"{percentile:.0f}th")
        else:
            st.metric("Percentile Rank", "N/A")
    
    with col3:
        st.metric("Expense Ratio", f"{fund_data['Net Expense Ratio']:.4f}%")
    
    with col4:
        aum = fund_data['Share Class Assets Under Management']
        st.metric("AUM", format_currency(aum))
    
    st.markdown("---")
    
    # Detailed metrics table
    st.subheader("📋 Detailed Metrics")
    
    metrics_data = []
    key_metrics = [
        ('Net Expense Ratio', 'Net Expense Ratio', '%'),
        ('Tracking Error (3Y)', 'Tracking Error (vs Category) (3Y)', ''),
        ('Tracking Error (5Y)', 'Tracking Error (vs Category) (5Y)', ''),
        ('Tracking Error (10Y)', 'Tracking Error (vs Category) (10Y)', ''),
        ('R² (5Y)', 'R-Squared (vs Category) (5Y)', ''),
        ('Max Drawdown (5Y)', 'Max Drawdown (5Y)', ''),
        ('Max Drawdown (10Y)', 'Max Drawdown (10Y)', ''),
        ('Downside Capture (5Y)', 'Downside (vs Category) (5Y)', ''),
        ('Info Ratio (3Y)', 'Information Ratio (vs Category) (3Y)', ''),
        ('Info Ratio (5Y)', 'Information Ratio (vs Category) (5Y)', ''),
        ('Sortino (3Y)', 'Historical Sortino (3Y)', ''),
        ('Sortino (5Y)', 'Historical Sortino (5Y)', ''),
        ('3Y Returns', '3 Year Total Returns (Daily)', ''),
        ('5Y Returns', '5 Year Total Returns (Daily)', ''),
        ('10Y Returns', '10 Year Total Returns (Daily)', ''),
    ]
    
    for display_name, col_name, unit in key_metrics:
        value = fund_data[col_name]
        if pd.notna(value):
            if unit == '%':
                formatted_value = f"{value:.4f}%"
            else:
                formatted_value = f"{value:.4f}"
            metrics_data.append({
                'Metric': display_name,
                'Value': formatted_value,
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Gauge chart
    st.subheader("📈 Score Gauge")
    
    if pd.notna(score):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 100], 'y': [0, 100]},
            title={'text': "Overall Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': SCORE_COLORS['poor']},
                    {'range': [40, 60], 'color': SCORE_COLORS['fair']},
                    {'range': [60, 80], 'color': SCORE_COLORS['good']},
                    {'range': [80, 100], 'color': SCORE_COLORS['excellent']},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: MULTI-FUND COMPARISON
# ============================================================================

def page_multi_comparison(df):
    """Multi-fund comparison page"""
    st.title("⚖️ Multi-Fund Comparison")
    st.write("Compare 2-5 funds side-by-side")
    st.markdown("---")
    
    # Select funds
    available_symbols = df['Symbol'].values
    selected_symbols = st.multiselect(
        "Select 2-5 funds to compare",
        available_symbols,
        max_selections=5,
        format_func=lambda x: f"{x} - {df[df['Symbol']==x]['Name'].values[0]}"
    )
    
    if len(selected_symbols) < 2:
        st.warning("Please select at least 2 funds to compare")
        return
    
    comparison_df = df[df['Symbol'].isin(selected_symbols)]
    
    st.markdown("---")
    
    # Comparison table
    st.subheader("📊 Score Comparison")
    
    comparison_table = comparison_df[[
        'Symbol', 'Name', 'Category Name', 'Fund_Type', 'Score', 
        'Percentile', 'Net Expense Ratio'
    ]].copy()
    
    comparison_table.columns = [
        'Ticker', 'Name', 'Category', 'Type', 'Score', 'Percentile', 'Expense %'
    ]
    
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    
    # Bar chart comparison
    st.subheader("📈 Score Comparison Chart")
    
    fig = px.bar(
        comparison_df.sort_values('Score', ascending=False),
        x='Symbol',
        y='Score',
        color='Score',
        color_continuous_scale=[[0, SCORE_COLORS['poor']], 
                               [0.4, SCORE_COLORS['fair']], 
                               [0.6, SCORE_COLORS['good']], 
                               [0.8, SCORE_COLORS['excellent']]],
        labels={'Score': 'Fund Score (0-100)', 'Symbol': 'Fund Ticker'},
        height=400
    )
    fig.update_yaxis(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics comparison
    st.subheader("📋 Detailed Metrics Comparison")
    
    metrics_comparison = comparison_df[[
        'Symbol', 'Net Expense Ratio', 'Tracking Error (vs Category) (5Y)',
        'Max Drawdown (5Y)', 'Information Ratio (vs Category) (5Y)',
        'Historical Sortino (5Y)', '5 Year Total Returns (Daily)'
    ]].copy()
    
    st.dataframe(metrics_comparison, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 3: PEER RANKINGS
# ============================================================================

def page_peer_rankings(df):
    """Peer rankings page"""
    st.title("🏆 Peer Rankings")
    st.write("See how funds rank within their categories")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox(
            "Select Category",
            sorted(df['Category Name'].unique())
        )
    
    with col2:
        selected_type = st.selectbox(
            "Fund Type",
            ["All", "Passive", "Active"]
        )
    
    # Filter data
    filtered_df = df[df['Category Name'] == selected_category].copy()
    
    if selected_type == "Passive":
        filtered_df = filtered_df[filtered_df['Index Fund'] == True]
    elif selected_type == "Active":
        filtered_df = filtered_df[filtered_df['Index Fund'] == False]
    
    filtered_df = filtered_df.dropna(subset=['Score'])
    filtered_df = filtered_df.sort_values('Score', ascending=False)
    
    st.markdown("---")
    
    # Category stats
    st.subheader("📊 Category Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Funds", len(filtered_df))
    with col2:
        avg_score = filtered_df['Score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    with col3:
        st.metric("Top Score", f"{filtered_df['Score'].max():.1f}")
    with col4:
        st.metric("Bottom Score", f"{filtered_df['Score'].min():.1f}")
    
    st.markdown("---")
    
    # Rankings table
    st.subheader("📋 Full Rankings")
    
    rankings_table = filtered_df[[
        'Symbol', 'Name', 'Score', 'Percentile', 'Net Expense Ratio',
        '5 Year Total Returns (Daily)'
    ]].copy()
    
    rankings_table['Rank'] = range(1, len(rankings_table) + 1)
    rankings_table = rankings_table[['Rank', 'Symbol', 'Name', 'Score', 'Percentile', 'Net Expense Ratio']]
    rankings_table.columns = ['Rank', 'Ticker', 'Name', 'Score', 'Percentile', 'Expense %']
    
    st.dataframe(rankings_table, use_container_width=True, hide_index=True)
    
    # Top 10 chart
    st.subheader("🥇 Top 10 Funds")
    
    top_10 = filtered_df.head(10).sort_values('Score', ascending=True)
    fig = px.barh(
        top_10,
        x='Score',
        y='Symbol',
        color='Score',
        color_continuous_scale='Greens',
        labels={'Score': 'Score', 'Symbol': 'Ticker'},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution histogram
    st.subheader("📈 Score Distribution")
    
    fig = px.histogram(
        filtered_df,
        x='Score',
        nbins=20,
        labels={'Score': 'Fund Score'},
        height=400,
        color_discrete_sequence=['#0066cc']
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: CATEGORY OVERVIEW
# ============================================================================

def page_category_overview(df):
    """Category overview page"""
    st.title("📊 Category Overview")
    st.write("Comprehensive analysis by category")
    st.markdown("---")
    
    selected_category = st.selectbox(
        "Select Category",
        sorted(df['Category Name'].unique())
    )
    
    category_df = df[df['Category Name'] == selected_category].copy()
    
    st.markdown("---")
    
    # Category stats
    st.subheader("📊 Category Statistics")
    
    valid_scores = category_df['Score'].dropna()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Funds", len(category_df))
    with col2:
        st.metric("Avg Score", f"{valid_scores.mean():.1f}" if len(valid_scores) > 0 else "N/A")
    with col3:
        st.metric("Median Score", f"{valid_scores.median():.1f}" if len(valid_scores) > 0 else "N/A")
    with col4:
        st.metric("Min Score", f"{valid_scores.min():.1f}" if len(valid_scores) > 0 else "N/A")
    with col5:
        st.metric("Max Score", f"{valid_scores.max():.1f}" if len(valid_scores) > 0 else "N/A")
    
    st.markdown("---")
    
    # Top performers
    st.subheader("🥇 Top Performers")
    
    top_performers = category_df.dropna(subset=['Score']).nlargest(5, 'Score')[[
        'Symbol', 'Name', 'Score', 'Net Expense Ratio', 'Fund_Type'
    ]]
    
    st.dataframe(top_performers, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Bottom performers
    st.subheader("📉 Lowest Scores")
    
    bottom_performers = category_df.dropna(subset=['Score']).nsmallest(5, 'Score')[[
        'Symbol', 'Name', 'Score', 'Net Expense Ratio', 'Fund_Type'
    ]]
    
    st.dataframe(bottom_performers, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Distribution chart
    st.subheader("📈 Score Distribution")
    
    fig = px.histogram(
        category_df.dropna(subset=['Score']),
        x='Score',
        nbins=15,
        title=f"Score Distribution - {selected_category}",
        labels={'Score': 'Fund Score'},
        height=400,
        color_discrete_sequence=['#2E86DE']
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
