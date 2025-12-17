# Fund Scoring System 2025 - Streamlit Application

This is a professional-grade Streamlit application for scoring 4,000+ mutual funds and ETFs using dual-framework scoring systems (passive and active funds).

## Installation

```bash
pip install streamlit pandas numpy scipy plotly reportlab pillow
```

## Files Included

1. **app.py** - Main Streamlit application (multi-page)
2. **config.py** - Configuration file with weights and settings
3. **scoring_engine.py** - Core scoring logic for passive and active funds
4. **pdf_generator.py** - PDF scorecard generation
5. **fund_screener_data-41.csv** - Fund data (4,685 funds)

## Running the Application

```bash
streamlit run app.py
```

## Features

### Dashboard Pages

1. **Single Fund Lookup** - Search and detailed analysis of individual funds
2. **Multi-Fund Comparison** - Compare 2-5 funds side-by-side with visualizations
3. **Peer Rankings** - Category-based rankings with top/bottom performers
4. **Category Overview** - Category statistics and distribution analysis

### PDF Export

Generate professional scorecards with:
- Overall score and percentile ranking
- Detailed metric breakdown by category
- Comparative visualizations (gauge, bar, radar charts)
- Peer comparison analysis

## Scoring Methodology

### Passive Fund Scoring (40 Funds Weighted by Category)

- **Cost (40%)**: Net Expense Ratio
- **Tracking Accuracy (33%)**: 3Y/5Y/10Y Tracking Error
- **Structural Fidelity (5%)**: R² 5Y
- **Stability (6%)**: Share Class AUM
- **Risk Management (16%)**: Downside Capture, Max Drawdown

### Active Fund Scoring (3,641 Funds)

- **Cost (25%)**: Net Expense Ratio
- **Skill (40%)**: Information Ratio, Sortino Ratio (3Y/5Y/10Y)
- **Risk Management (20%)**: Max Drawdown, Downside Capture
- **Performance (15%)**: Returns, Upside Capture

## Data Handling

- Missing data handled gracefully with proportional weight rescaling
- Funds with >30% missing data flagged as "incomplete"
- All scores calculated within fund type category (Passive/Active)

## Performance

- Dashboard loads in <3 seconds
- Handles 4,685 funds efficiently
- Cached data processing for optimal performance

## Testing Data

Verify calculations using these funds:

**Passive Funds:**
- VOO (Vanguard S&P 500 ETF) - Expected: ~95-99 score
- IWD (iShares Russell 1000 Value) - Expected: ~80-90 score
- SCHD (Schwab U.S. Dividend Equity ETF) - Expected: ~60-75 score

**Active Funds:**
- PEIYX (Putnam Large Cap Value Y) - Expected: ~70-85 score
- PRBLX (Parnassus Core Equity) - Expected: ~50-65 score
- ABVYX (AB Large Cap Value Fund Adv) - Expected: ~60-75 score

## Support

For issues or customization requests, contact the development team.
