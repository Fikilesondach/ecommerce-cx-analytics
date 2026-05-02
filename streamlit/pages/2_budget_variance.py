"""
2_budget_variance.py
--------------------
Streamlit page: Budget Variance Analysis
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.generate_data import generate_all_data
from src.budget.variance_analysis import run_variance_analysis

st.set_page_config(page_title="Budget Variance", page_icon="💸", layout="wide")
st.title("💸 Budget Variance Analysis")
st.caption("Decomposing the gap between budget and actual into volume, rate, and mix components")
st.markdown("---")

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_data():
    return generate_all_data(save=False)

@st.cache_data(show_spinner="Running variance analysis...")
def load_variance(costs_df):
    return run_variance_analysis(costs_df=costs_df)

datasets         = load_data()
variance_results = load_variance(datasets['costs'])
summary          = variance_results['brand_summary']

# ---------------------------------------------------------------------------
# TOP-LINE METRICS
# ---------------------------------------------------------------------------

total_budget   = summary['budget_cost'].sum()
total_actual   = summary['actual_cost'].sum()
total_variance = summary['total_variance'].sum()
var_pct        = total_variance / total_budget * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Budget",   f"R{total_budget/1e6:.2f}M")
col2.metric("Total Actual",   f"R{total_actual/1e6:.2f}M")
col3.metric("Total Variance", f"+R{total_variance/1e3:.0f}K", f"+{var_pct:.1f}%")
col4.metric("Primary Driver", "Volume", "99%+ of variance")

st.markdown("---")

# ---------------------------------------------------------------------------
# VARIANCE DECOMPOSITION TABLE
# ---------------------------------------------------------------------------

st.subheader("Variance Decomposition by Brand (Annual)")
st.caption("Positive = over budget (unfavourable) | Negative = under budget (favourable)")

display = summary[[
    'brand', 'budget_cost', 'actual_cost', 'total_variance',
    'volume_variance', 'rate_variance', 'mix_variance', 'variance_pct'
]].copy()

display.columns = [
    'Brand', 'Budget (R)', 'Actual (R)', 'Total Var (R)',
    'Volume Var (R)', 'Rate Var (R)', 'Mix Var (R)', '% Over'
]
for col in ['Budget (R)', 'Actual (R)', 'Total Var (R)',
            'Volume Var (R)', 'Rate Var (R)', 'Mix Var (R)']:
    display[col] = display[col].apply(lambda x: f"R{x:,.0f}")
display['% Over'] = display['% Over'].apply(lambda x: f"+{x:.1f}%")

st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# WATERFALL CHARTS
# ---------------------------------------------------------------------------

st.subheader("Waterfall Charts")

brands   = sorted(summary['brand'].unique())
col_tabs = st.tabs(brands + ["Cross-Brand"])

for i, brand in enumerate(brands):
    with col_tabs[i]:
        path = Path(f"outputs/figures/variance_waterfall_{brand.lower()}.png")
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning("Run main.py first to generate figures.")

with col_tabs[-1]:
    path = Path("outputs/figures/variance_cross_brand.png")
    if path.exists():
        st.image(str(path), use_container_width=True)
    path2 = Path("outputs/figures/variance_monthly_trend.png")
    if path2.exists():
        st.image(str(path2), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------------------------

with st.expander("Methodology — Variance Decomposition"):
    st.markdown("""
The total variance (actual cost − budget cost) is split into three independent components:

**Volume Variance**
How much of the overrun is explained purely by receiving more contacts than budgeted?

```
Volume Variance = (Actual Hours − Budget Hours) × Budget Rate
```

**Rate Variance**
How much is explained by cost per hour moving versus the budgeted rate?

```
Rate Variance = (Actual Rate − Budget Rate) × Actual Hours
```

**Mix Variance**
The residual — what is not explained by volume or rate alone. In a real business,
this captures channel mix shifts. If more contacts came through phone than budgeted,
average cost rises even if individual channel rates held.

```
Mix Variance = Total Variance − Volume Variance − Rate Variance
```

In this simulation, volume variance accounts for >99% of total overrun because
the budget deliberately underestimated contact volumes by 3–10%. Rate and mix
variance are negligible, confirming the cost model is internally consistent.
    """)
