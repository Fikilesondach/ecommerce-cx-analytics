"""
1_forecasting.py
----------------
Streamlit page: Contact Volume Forecasting
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.generate_data import generate_all_data
from src.forecasting.time_series import run_forecasting

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
st.title("🔮 Contact Volume Forecasting")
st.caption("Seasonal decomposition + Holt-Winters triple exponential smoothing")
st.markdown("---")

# ---------------------------------------------------------------------------
# LOAD DATA (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_data():
    return generate_all_data(save=False)

@st.cache_data(show_spinner="Running forecasting models...")
def load_forecast(contacts_df):
    return run_forecasting(contacts_df=contacts_df)

datasets         = load_data()
forecast_results = load_forecast(datasets['contacts'])

# ---------------------------------------------------------------------------
# MAPE SUMMARY
# ---------------------------------------------------------------------------

st.subheader("Model Accuracy — MAPE by Brand")
st.caption("Mean Absolute Percentage Error on in-sample validation (final 6 months). Lower is better.")

summary = forecast_results['summary']
cols    = st.columns(len(summary))

for col, (_, row) in zip(cols, summary.iterrows()):
    delta_color = "normal" if row['MAPE (%)'] < 10 else "inverse"
    col.metric(
        label=row['Brand'],
        value=f"{row['MAPE (%)']:.1f}%",
        delta="✓ Below 10% benchmark",
        delta_color="normal"
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# FORECAST TABLE
# ---------------------------------------------------------------------------

st.subheader("6-Month Forecast (January – June 2024)")

display_summary = summary.copy()
month_cols = [c for c in display_summary.columns if 'Fcst' in c]
display_summary[month_cols] = display_summary[month_cols].applymap(lambda x: f"{int(x):,}")
display_summary['MAPE (%)'] = display_summary['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# CHARTS — one per brand
# ---------------------------------------------------------------------------

st.subheader("Forecast Charts by Brand")
st.caption("Solid line = historical actuals | Dashed = model fit | Green dashed = future forecast")

brands   = sorted(datasets['contacts']['brand'].unique())
selected = st.selectbox("Select brand", brands)

fig_path = Path(f"outputs/figures/forecast_holtwinters_{selected.lower()}.png")
dec_path = Path(f"outputs/figures/forecast_decomposition_{selected.lower()}.png")

tab1, tab2 = st.tabs(["Holt-Winters Forecast", "Seasonal Decomposition"])

with tab1:
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.warning("Run the pipeline first to generate figures.")

with tab2:
    if dec_path.exists():
        st.image(str(dec_path), use_container_width=True)
        st.caption(
            "Decomposition separates the series into trend, seasonal, and residual components. "
            "A stable seasonal component (consistent shape year over year) justifies using "
            "Holt-Winters, which models trend and seasonality simultaneously."
        )
    else:
        st.warning("Run the pipeline first to generate figures.")

st.markdown("---")

# ---------------------------------------------------------------------------
# METHODOLOGY NOTE
# ---------------------------------------------------------------------------

with st.expander("Methodology"):
    st.markdown("""
**Stage 1 — Seasonal Decomposition (diagnostic)**

`statsmodels.tsa.seasonal.seasonal_decompose` separates the contact volume series into:
- **Trend** — the underlying direction of growth
- **Seasonal** — the repeating annual pattern (November/December spikes)
- **Residual** — unexplained noise

This is a diagnostic step, not a forecast. It confirms the seasonal pattern is stable
enough to model with Holt-Winters.

**Stage 2 — Holt-Winters Exponential Smoothing (forecast)**

`statsmodels.tsa.holtwinters.ExponentialSmoothing` with:
- `trend='add'` — linear trend component
- `seasonal='add'` — additive seasonality (spike size is constant, not proportional)
- `seasonal_periods=12` — monthly data, annual cycle
- `initialization_method='estimated'` — parameters optimised by MLE

**Why additive?** The November/December spike is roughly constant in absolute size
year over year, rather than growing proportionally with the trend. Additive decomposition
reflects this correctly.

**Why in-sample validation?** Holt-Winters requires at least two full seasonal cycles
(24 months) to initialise. A train/test split on 24 months leaves only 18 months in
training — insufficient. In-sample MAPE on the final 6 months is the methodologically
correct response to a short series.
    """)
