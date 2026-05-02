"""
app.py
------
Streamlit entry point for ecommerce-cx-analytics.

Run with:
    streamlit run streamlit/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Make src/ importable from the streamlit/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="eCommerce CX Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.title("📊 CX Analytics")
st.sidebar.caption("Multi-Brand Contact Centre | Jan 2022 – Dec 2023")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Modules**
- 🔮 Forecasting
- 💸 Budget Variance
- 📦 Cost Allocation
- 👥 Capacity Planning
- 🗂️ Executive Dashboard
""")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · statsmodels · Streamlit")

# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------

st.title("eCommerce CX Analytics")
st.subheader("Multi-Brand Contact Centre Analytics & Forecasting Engine")

st.markdown("""
A complete analytical system for a simulated South African e-commerce group
operating four brands across a shared contact centre.
""")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Contacts", "299,017", "24 months")
with col2:
    st.metric("Budget Variance", "+9.6%", "over budget")
with col3:
    st.metric("Avg CSAT", "7.62 / 10", "target: 7.5")
with col4:
    st.metric("Forecast MAPE", "<1.1%", "all brands")

st.markdown("---")

st.markdown("### What This System Does")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
**🔮 Contact Volume Forecasting**
Seasonal decomposition + Holt-Winters triple exponential smoothing.
Trained per brand, validated on held-out months, forecasts 6 months ahead.

**💸 Budget Variance Analysis**
Decomposes the gap between budget and actual into volume, rate, and mix components.
Waterfall charts per brand plus cross-brand executive view.
""")

with col_b:
    st.markdown("""
**📦 Cost Allocation**
Distributes shared service costs across brands using two methods:
simple volume weighting and channel-adjusted weighting.

**👥 Capacity Planning**
Links forecast contact volumes to agent headcount requirements.
Three scenarios: Base, Growth (+20%), Stress (+40%).
""")

st.markdown("---")
st.markdown("### Brands")

b1, b2, b3, b4 = st.columns(4)
with b1:
    st.info("**Takealot.com**\nHigh volume · Broad product mix")
with b2:
    st.warning("**Mr D**\nFood delivery · Highest frequency")
with b3:
    st.success("**Sellers**\nB2B · Lower volume · Higher complexity")
with b4:
    st.error("**TFS**\nTakealot Fulfilment · Logistics contacts")

st.markdown("---")
st.caption("Navigate using the sidebar pages. All modules run live — no pre-cached results.")
