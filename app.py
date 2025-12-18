import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
from datetime import datetime, timedelta
import glob
import os
import json
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Fund Screening System 2025", layout="wide")

VERSIONS_DIR = "versions"
METADATA_FILE = os.path.join(VERSIONS_DIR, "metadata.json")
os.makedirs(VERSIONS_DIR, exist_ok=True)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def load_csv():
    """Load CSV without caching - always fresh data"""
    csvfiles = glob.glob("*.csv")
    if csvfiles:
        try:
            df = pd.read_csv(csvfiles[0])
            return df, csvfiles[0]
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None, None
    return None, None

def load_previous_version(version_num):
    """Load a specific version from archive"""
    metadata = load_metadata()
    if f"v{version_num}" in metadata:
        filepath = os.path.join(VERSIONS_DIR, metadata[f"v{version_num}"]["filename"])
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
    return None

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
    
    # Calculate metrics for metadata
    avg_score = 0
    if "Score" in df.columns:
        score_vals = pd.to_numeric(df["Score"], errors='coerce').dropna()
        avg_score = float(score_vals.mean()) if len(score_vals) > 0 else 0
    
    avg_percentile = 0
    if "Percentile_Calculated" in df.columns:
        percentile_vals = pd.to_numeric(df["Percentile_Calculated"], errors='coerce').dropna()
        avg_percentile = float(percentile_vals.mean()) if len(percentile_vals) > 0 else 0
    
    metadata[f"v{version_num}"] = {
        "filename": filename,
        "date": date_str,
        "timestamp": datetime.now().isoformat(),
        "totalFunds": len(df),
        "avgScore": avg_score,
        "avgPercentile": avg_percentile,
        "notes": notes
    }
    save_metadata(metadata)
    st.success(f"Saved as Version {version_num}")
    return version_num

def get_data_quality_metrics(df):
    """Analyze data quality across the dataset"""
    metrics = {}
    
    for col in df.columns:
        total = len(df)
        missing = df[col].isna().sum()
        pct_complete = ((total - missing) / total * 100) if total > 0 else 0
        metrics[col] = {
            "missing": missing,
            "pct_complete": pct_complete,
            "pct_missing": 100 - pct_complete
        }
    
    return metrics

def get_funds_affected_by_nulls(df, threshold=30):
    """Find funds most affected by missing data"""
    missing_per_fund = df.isna().sum(axis=1)
    total_cols = len(df.columns)
    missing_pct = (missing_per_fund / total_cols * 100)
    
    affected = pd.DataFrame({
        "Symbol": df["Symbol"],
        "Name": df.get("Name", "N/A"),
        "Category": df.get("Category Name", "N/A"),
        "Missing_Fields": missing_per_fund,
        "Missing_Pct": missing_pct
    })
    
    affected = affected[affected["Missing_Pct"] > threshold].sort_values("Missing_Pct", ascending=False)
    return affected

def calculate_score_changes(current_df, previous_df):
    """Calculate score changes between versions"""
    merged = current_df.merge(
        previous_df[["Symbol", "Score"]],
        on="Symbol",
        how="left",
        suffixes=("_current", "_previous")
    )
    
    merged["Score_Change"] = merged["Score_current"] - merged["Score_previous"]
    
    return merged

def categorize_fund_status(score_change):
    """Categorize fund status based on score change"""
    if pd.isna(score_change):
        return "🆕 NEW", "blue"
    elif score_change >= 5:
        return "✅ IMPROVING ↑", "green"
    elif score_change < -10:
        return "🚩 ALERT ↓↓", "red"
    elif score_change < -5:
        return "⚠️ DECLINING ↓", "orange"
    else:
        return "STABLE", "gray"

df, csvname = load_csv()

if df is None:
    st.title("Fund Screening System 2025")
    st.markdown("---")
    st.write("No CSV file found in directory.")
    st.write(f"Current directory: {os.getcwd()}")
    st.write("Expected: A CSV file exported from Excel scoring template with columns including 'Symbol', 'Score', 'Percentile_Calculated', 'Category Name', etc.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"✅ Loaded {len(df)} funds")
            
            required_cols = ["Symbol", "Score", "Percentile_Calculated"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.warning(f"⚠️ Missing columns: {', '.join(missing)}")
                st.write(f"Available columns: {list(df.columns)}")
            
            notes = st.text_input("Add notes for this version (optional)", "")
            if st.button("Save as New Version"):
                version_num = save_version(df, notes)
                st.info(f"✅ Version {version_num} saved! Please refresh the page to see the updated app.")
                st.balloons()
                st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

def get_fund_score(symbol):
    """Get score for a fund"""
    fund_row = df[df["Symbol"] == symbol]
    if not fund_row.empty:
        score = fund_row.iloc[0].get("Score", None)
        if score is not None:
            try:
                return float(score)
            except (ValueError, TypeError):
                return None
    return None

def get_fund_percentile(symbol):
    """Get percentile for a fund"""
    fund_row = df[df["Symbol"] == symbol]
    if not fund_row.empty:
        percentile = fund_row.iloc[0].get("Percentile_Calculated", None)
        if percentile is not None:
            try:
                return float(percentile)
            except (ValueError, TypeError):
                return None
    return None

def get_score_label(score):
    """Convert score to human-readable label"""
    if score is None:
        return "N/A"
    elif score >= 80:
        return "🟢 Excellent"
    elif score >= 70:
        return "🟡 Good"
    elif score >= 60:
        return "🟠 Fair"
    else:
        return "🔴 Below Average"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", 
    ["Single Fund Lookup", "Multi-Fund Comparison", "Peer Rankings", 
     "Data Quality Dashboard", "Score Volatility Monitor", "Category Health",
     "Upload New Version", "Version History", "IC Committee Export"])

if page == "Single Fund Lookup":
    st.title("Single Fund Lookup")
    st.markdown("---")
    st.write("**Methodology:** Rankings based on custom scoring by category with percentile rank.")
    
    search_symbol = st.text_input("Enter fund symbol", placeholder="e.g., AVLC, PRILX").upper()
    
    if search_symbol:
        fund = df[df["Symbol"] == search_symbol]
        
        if not fund.empty:
            fund = fund.iloc[0]
            score = get_fund_score(search_symbol)
            percentile = get_fund_percentile(search_symbol)
            category = fund.get("Category Name", "N/A")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Symbol", search_symbol)
            with col2:
                st.metric("Name", fund.get("Name", "N/A")[:30])
            with col3:
                st.metric("Category", category[:25])
            with col4:
                if score:
                    st.metric(f"Score", f"{score:.0f}/100")
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Fund Details")
                details_data = {
                    "Metric": ["Performance Level", "Category", "Fund Type", "Expense Ratio", "5-Year Return"],
                    "Value": [
                        get_score_label(score),
                        category,
                        "Index Fund" if fund.get("Index Fund", False) else "Active Fund",
                        f"{fund.get('Net Expense Ratio', 0)*100:.4f}%" if "Net Expense Ratio" in fund else "N/A",
                        f"{fund.get('5 Year Total Returns (Daily)', 0)*100:.2f}%" if "5 Year Total Returns (Daily)" in fund else "N/A",
                    ]
                }
                st.dataframe(pd.DataFrame(details_data), use_container_width=True)
                
                st.subheader("Performance Returns")
                returns_data = {
                    "Period": ["3-Year", "5-Year", "10-Year"],
                    "Return": [
                        f"{fund.get('3 Year Total Returns (Daily)', 0)*100:.2f}%" if "3 Year Total Returns (Daily)" in fund and pd.notna(fund.get("3 Year Total Returns (Daily)")) else "N/A",
                        f"{fund.get('5 Year Total Returns (Daily)', 0)*100:.2f}%" if "5 Year Total Returns (Daily)" in fund and pd.notna(fund.get("5 Year Total Returns (Daily)")) else "N/A",
                        f"{fund.get('10 Year Total Returns (Daily)', 0)*100:.2f}%" if "10 Year Total Returns (Daily)" in fund and pd.notna(fund.get("10 Year Total Returns (Daily)")) else "N/A",
                    ]
                }
                st.dataframe(pd.DataFrame(returns_data), use_container_width=True)
            
            with col2:
                st.subheader("Scoring Breakdown")
                st.metric("Score", f"{score:.0f}/100")
                st.metric("Percentile Rank", f"{percentile:.0f}%" if percentile else "N/A")
                st.caption("*Percentile calculated by category rank*")
        else:
            st.error(f"❌ Fund {search_symbol} not found")
            st.write(f"Try one of these: {', '.join(df['Symbol'].head(10).tolist())}")

elif page == "Multi-Fund Comparison":
    st.title("Multi-Fund Comparison - IC Committee Review")
    st.markdown("---")
    st.write("Compare funds side-by-side with return data, scores, and risk flags.")
    
    categories = sorted(df["Category Name"].dropna().unique()) if "Category Name" in df.columns else []
    selected_category = st.selectbox("Filter by Category", ["All"] + categories if categories else ["All"])
    
    filtered_df = df if selected_category == "All" or not categories else df[df["Category Name"] == selected_category]
    symbols = st.multiselect("Select funds to compare", sorted(filtered_df["Symbol"].unique()), max_selections=10)
    
    if symbols:
        comparison_data = []
        for symbol in symbols:
            fund = df[df["Symbol"] == symbol].iloc[0]
            score = get_fund_score(symbol)
            percentile = get_fund_percentile(symbol)
            
            flags = []
            if pd.notna(fund.get("Status")) and fund.get("Status") == "RED":
                flags.append("🚩 RED FLAG")
            elif pd.notna(fund.get("Status")) and fund.get("Status") == "YELLOW":
                flags.append("⚠️ DECLINING")
            
            comparison_data.append({
                "Symbol": symbol,
                "Name": fund.get("Name", "N/A")[:35],
                "Score": f"{score:.0f}" if score else "N/A",
                "Percentile": f"{percentile:.0f}%" if percentile else "N/A",
                "5Y Return": f"{fund.get('5 Year Total Returns (Daily)', 0)*100:.2f}%" if "5 Year Total Returns (Daily)" in fund else "N/A",
                "Expense Ratio": f"{fund.get('Net Expense Ratio', 0)*100:.3f}%" if "Net Expense Ratio" in fund else "N/A",
                "Flags": " ".join(flags) if flags else "✅ Clean",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, height=400)

elif page == "Peer Rankings":
    st.title("Peer Rankings by Category")
    st.markdown("---")
    
    categories = sorted(df["Category Name"].dropna().unique()) if "Category Name" in df.columns else []
    selected_category = st.selectbox("Select Category", categories if categories else [])
    
    if selected_category:
        category_df = df[df["Category Name"] == selected_category].copy()
        category_df["Score"] = category_df["Symbol"].apply(get_fund_score)
        category_df = category_df.sort_values("Score", ascending=False, na_position="last")
        
        st.subheader(f"🏆 Top 10 Performers in {selected_category}")
        top10 = category_df.head(10)[["Symbol", "Name", "Score"]].copy()
        top10["Score"] = top10["Score"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        st.dataframe(top10, use_container_width=True)
        
        st.subheader("Score Distribution")
        scores = category_df["Score"].dropna()
        fig = go.Figure(data=go.Histogram(x=scores, nbinsx=15, marker=dict(color="1E80A0")))
        fig.update_layout(
            title=f"Score Distribution - {selected_category}",
            xaxis_title="Score (0-100)",
            yaxis_title="Number of Funds",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Data Quality Dashboard":
    st.title("📊 Data Quality Dashboard")
    st.markdown("---")
    st.write("Monitor data completeness and identify funds with significant missing data.")
    
    metrics = get_data_quality_metrics(df)
    
    st.subheader("Column Data Completeness")
    col_stats = pd.DataFrame([
        {"Column": col, "Complete": metrics[col]["pct_complete"], "Missing": metrics[col]["pct_missing"]}
        for col in sorted(metrics.keys(), key=lambda x: metrics[x]["pct_complete"])
    ])
    
    fig = go.Figure(data=go.Bar(
        y=col_stats["Column"],
        x=col_stats["Complete"],
        orientation="h",
        marker=dict(color="1E80A0")
    ))
    fig.update_layout(
        title="Data Completeness by Column",
        xaxis_title="% Complete",
        height=600,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Funds Most Affected by Missing Data")
    affected = get_funds_affected_by_nulls(df, threshold=30)
    if not affected.empty:
        st.dataframe(affected.head(20), use_container_width=True)
        st.caption(f"Showing {len(affected)} funds with >30% missing fields")
    else:
        st.success("✅ No funds with significant missing data (>30%)")
    
    st.subheader("Data Quality by Category")
    category_quality = []
    for category in df["Category Name"].unique():
        cat_df = df[df["Category Name"] == category]
        total_fields = len(cat_df) * len(cat_df.columns)
        missing_fields = cat_df.isna().sum().sum()
        pct_complete = ((total_fields - missing_fields) / total_fields * 100) if total_fields > 0 else 0
        category_quality.append({
            "Category": category,
            "Funds": len(cat_df),
            "Data_Complete": f"{pct_complete:.1f}%",
            "Missing_Fields": missing_fields
        })
    
    quality_df = pd.DataFrame(category_quality).sort_values("Data_Complete")
    st.dataframe(quality_df, use_container_width=True)

elif page == "Score Volatility Monitor":
    st.title("⚡ Score Volatility Monitor")
    st.markdown("---")
    st.write("Track fund score changes between versions and identify funds requiring attention.")
    
    metadata = load_metadata()
    if len(metadata) < 2:
        st.info("Need at least 2 versions to compare. Upload another version to enable this feature.")
    else:
        versions = sorted(metadata.keys(), key=lambda x: int(x.replace("v", "")))
        
        col1, col2 = st.columns(2)
        with col1:
            from_version = st.selectbox("From Version", versions[:-1])
        with col2:
            to_version = st.selectbox("To Version", versions[versions.index(from_version)+1:], index=len(versions)-2 if len(versions) > 1 else 0)
        
        if from_version and to_version:
            prev_df = load_previous_version(int(from_version.replace("v", "")))
            curr_df = load_previous_version(int(to_version.replace("v", ""))) if to_version != f"v{len(metadata)}" else df
            
            if prev_df is not None and curr_df is not None:
                changes = calculate_score_changes(curr_df, prev_df)
                changes["Status"], changes["Status_Color"] = zip(*changes["Score_Change"].apply(lambda x: categorize_fund_status(x)))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    new_funds = len(changes[changes["Status"] == "🆕 NEW"])
                    st.metric("New Funds", new_funds)
                with col2:
                    improving = len(changes[changes["Status"] == "✅ IMPROVING ↑"])
                    st.metric("Improving", improving, delta=improving)
                with col3:
                    declining = len(changes[changes["Status"] == "⚠️ DECLINING ↓"])
                    st.metric("Declining", declining, delta=-declining)
                with col4:
                    alerts = len(changes[changes["Status"] == "🚩 ALERT ↓↓"])
                    st.metric("🚩 ALERTS", alerts, delta=-alerts)
                
                st.markdown("---")
                
                st.subheader("Score Changes (Sorted by Largest Decline)")
                changes_sorted = changes.dropna(subset=["Score_Change"]).sort_values("Score_Change")
                
                display_cols = ["Symbol", "Name", "Category Name", "Score_previous", "Score_current", "Score_Change", "Status"]
                display_df = changes_sorted[display_cols].copy()
                display_df.columns = ["Symbol", "Fund Name", "Category", f"Score ({from_version})", f"Score ({to_version})", "Change", "Status"]
                
                st.dataframe(display_df, use_container_width=True, height=500)

elif page == "Category Health":
    st.title("📈 Category Health Scorecard")
    st.markdown("---")
    
    category_stats = []
    for category in df["Category Name"].unique():
        cat_df = df[df["Category Name"] == category]
        scores = pd.to_numeric(cat_df["Score"], errors='coerce').dropna()
        
        category_stats.append({
            "Category": category,
            "# Funds": len(cat_df),
            "Avg Score": f"{scores.mean():.0f}" if len(scores) > 0 else "N/A",
            "Best Score": f"{scores.max():.0f}" if len(scores) > 0 else "N/A",
            "Worst Score": f"{scores.min():.0f}" if len(scores) > 0 else "N/A",
            "Data Quality": f"{(100-cat_df.isna().sum().sum()/(len(cat_df)*len(cat_df.columns))*100):.0f}%"
        })
    
    health_df = pd.DataFrame(category_stats).sort_values("Avg Score", ascending=False, key=pd.to_numeric, errors='coerce')
    st.dataframe(health_df, use_container_width=True)

elif page == "Upload New Version":
    st.title("Upload New Version")
    st.markdown("---")
    st.write("Upload an updated CSV file from the Excel scoring template.")
    st.write("**Expected columns:** Symbol, Name, Category Name, Score, Percentile_Calculated, Status, etc.")
    
    uploaded_file = st.file_uploader("Upload CSV file from Excel template", type="csv")
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.write(f"✅ Loaded {len(new_df)} funds")
            
            required_cols = ["Symbol", "Score", "Percentile_Calculated"]
            missing = [col for col in required_cols if col not in new_df.columns]
            if missing:
                st.warning(f"⚠️ Missing columns: {', '.join(missing)}")
            
            notes = st.text_input("Add notes for this version (optional)", value="")
            
            if st.button("Save as New Version"):
                version_num = save_version(new_df, notes)
                st.info(f"✅ Version {version_num} saved successfully! Please refresh the page to see updates.")
                st.balloons()
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif page == "Version History":
    st.title("Version History & Tracking")
    st.markdown("---")
    
    metadata = load_metadata()
    
    if not metadata:
        st.info("ℹ️ No versions saved yet. Upload a CSV to start tracking!")
    else:
        st.subheader("All Versions")
        version_data = []
        
        for vkey in sorted(metadata.keys(), key=lambda x: int(x.replace("v", ""))):
            vinfo = metadata[vkey]
            version_data.append({
                "Version": vkey,
                "Date": vinfo["date"],
                "Total Funds": vinfo["totalFunds"],
                "Avg Score": f"{vinfo['avgScore']:.0f}" if vinfo.get('avgScore', 0) > 0 else "N/A",
                "Avg Percentile": f"{vinfo['avgPercentile']:.0f}%" if vinfo.get('avgPercentile', 0) > 0 else "N/A",
                "Notes": vinfo.get("notes", "")
            })
        
        version_df = pd.DataFrame(version_data)
        st.dataframe(version_df, use_container_width=True)
        
        if len(metadata) > 1:
            st.subheader("Average Score Trend")
            scores_over_time = []
            
            for vkey in sorted(metadata.keys(), key=lambda x: int(x.replace("v", ""))):
                vinfo = metadata[vkey]
                scores_over_time.append({
                    "Version": vkey,
                    "Date": vinfo["date"],
                    "Avg Score": vinfo["avgScore"]
                })
            
            trend_df = pd.DataFrame(scores_over_time)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df["Date"],
                y=trend_df["Avg Score"],
                mode="lines+markers",
                name="Avg Score",
                line=dict(color="1E80A0", width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Average Fund Score Over Time",
                xaxis_title="Date",
                yaxis_title="Average Score (0-100)",
                height=400,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Download Previous Versions")
        for vkey in sorted(metadata.keys(), reverse=True, key=lambda x: int(x.replace("v", ""))):
            vinfo = metadata[vkey]
            filename = vinfo["filename"]
            filepath = os.path.join(VERSIONS_DIR, filename)
            
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    st.download_button(
                        label=f"⬇️ {vkey} ({vinfo['date']})",
                        data=f.read(),
                        file_name=filename,
                        mime="text/csv"
                    )

elif page == "IC Committee Export":
    st.title("📋 IC Committee Export")
    st.markdown("---")
    st.write("Generate IC Committee reports by category with key comparison metrics.")
    
    categories = sorted(df["Category Name"].dropna().unique()) if "Category Name" in df.columns else []
    selected_categories = st.multiselect("Select categories for report", categories, default=categories[:3] if len(categories) > 0 else [])
    
    if selected_categories:
        st.subheader("Top Performers by Category")
        
        for category in selected_categories:
            cat_df = df[df["Category Name"] == category].copy()
            cat_df["Score"] = pd.to_numeric(cat_df["Score"], errors='coerce')
            top_funds = cat_df.nlargest(5, "Score")[["Symbol", "Name", "Score", "5 Year Total Returns (Daily)"]].copy()
            top_funds["Score"] = top_funds["Score"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
            top_funds["5Y Return"] = top_funds["5 Year Total Returns (Daily)"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
            
            st.write(f"**{category}**")
            st.dataframe(top_funds[["Symbol", "Name", "Score", "5Y Return"]], use_container_width=True)
            st.divider()
        
        if st.button("Generate PDF Report"):
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            story.append(Paragraph("IC Committee Fund Review Report", styles["Title"]))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
            story.append(Spacer(1, 0.3*inch))
            
            for category in selected_categories:
                cat_df = df[df["Category Name"] == category].copy()
                cat_df["Score"] = pd.to_numeric(cat_df["Score"], errors='coerce')
                top_funds = cat_df.nlargest(5, "Score")
                
                story.append(Paragraph(f"{category}", styles["Heading2"]))
                
                table_data = [["Symbol", "Fund Name", "Score", "5Y Return", "Expense Ratio"]]
                for _, fund in top_funds.iterrows():
                    table_data.append([
                        str(fund.get("Symbol", "N/A")),
                        str(fund.get("Name", "N/A"))[:40],
                        f"{fund.get('Score', 0):.0f}",
                        f"{fund.get('5 Year Total Returns (Daily)', 0)*100:.2f}%",
                        f"{fund.get('Net Expense Ratio', 0)*100:.3f}%"
                    ])
                
                table = Table(table_data, colWidths=[0.8*inch, 2.5*inch, 0.8*inch, 1*inch, 1*inch])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E80A0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey)
                ]))
                
                story.append(table)
                story.append(Spacer(1, 0.2*inch))
                story.append(PageBreak())
            
            doc.build(story)
            pdf_buffer.seek(0)
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"IC_Committee_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

st.markdown("---")
st.caption(f"📊 Data Source: {csvname} | Total Funds: {len(df)} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("🎯 **Methodology:** Funds scored 0-100 by category using custom weighting in Excel template. Percentile calculated within category peers.")