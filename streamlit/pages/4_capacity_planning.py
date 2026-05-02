"""
4_capacity_planning.py
----------------------
Streamlit page: Capacity Planning
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.generate_data import generate_all_data
from src.forecasting.time_series import run_forecasting
from src.budget.capacity_planning import run_capacity_planning

st.set_page_config(page_title="Capacity Planning", page_icon="👥", layout="wide")
st.title("👥 Capacity Planning")
st.caption("Linking forecast contact volumes to agent headcount — three scenarios")
st.markdown("---")

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_data():
    return generate_all_data(save=False)

@st.cache_data(show_spinner="Running forecasting...")
def load_forecast(contacts_df):
    return run_forecasting(contacts_df=contacts_df)

@st.cache_data(show_spinner="Building capacity plan...")
def load_capacity(forecast_summary):
    return run_capacity_planning(forecast_df=forecast_summary)

datasets         = load_data()
forecast_results = load_forecast(datasets['contacts'])
capacity_results = load_capacity(forecast_results['summary'])
capacity_df      = capacity_results['capacity_plan']
summary          = capacity_results['summary']

# ---------------------------------------------------------------------------
# ASSUMPTIONS PANEL
# ---------------------------------------------------------------------------

st.subheader("Model Assumptions")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Blended AHT",          "8.1 min",    "weighted avg across channels")
col2.metric("Productive Hrs/Agent", "132 hrs/mo", "22 days × 8hrs × 75%")
col3.metric("Shrinkage",            "25%",         "leave, training, breaks")
col4.metric("Cost per FTE",         "R28,000/mo", "fully loaded")

st.markdown("---")

# ---------------------------------------------------------------------------
# SCENARIO SELECTOR
# ---------------------------------------------------------------------------

st.subheader("Headcount Requirement by Scenario")

scenario = st.radio(
    "Select scenario",
    ["Base", "Growth (+20%)", "Stress (+40%)"],
    horizontal=True
)
scenario_map = {"Base": "Base", "Growth (+20%)": "Growth", "Stress (+40%)": "Stress"}
selected_scenario = scenario_map[scenario]

scenario_data = capacity_df[capacity_df['scenario'] == selected_scenario]
pivot = scenario_data.pivot_table(
    index='brand',
    columns='month',
    values='headcount'
)
pivot.columns = [m.strftime('%b %Y') for m in pivot.columns]
pivot['Peak HC'] = pivot.max(axis=1)
pivot['Peak Cost (R)'] = (pivot['Peak HC'] * 28000).apply(lambda x: f"R{x:,.0f}")
pivot['Peak HC'] = pivot['Peak HC'].astype(int)

st.dataframe(pivot, use_container_width=True)

# Total row
total_peak = scenario_data.groupby('month')['headcount'].sum().max()
st.info(f"**Total peak headcount across all brands — {selected_scenario} scenario: {int(total_peak)} FTE**  "
        f"(R{int(total_peak) * 28000:,.0f}/month)")

st.markdown("---")

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------

st.subheader("Scenario Comparison Charts")

tab1, tab2, tab3 = st.tabs([
    "By Brand",
    "Total Headcount",
    "Monthly Cost"
])

chart_map = {
    "By Brand":        "outputs/figures/capacity_scenario_by_brand.png",
    "Total Headcount": "outputs/figures/capacity_total_headcount.png",
    "Monthly Cost":    "outputs/figures/capacity_monthly_cost.png",
}

for tab, (label, path_str) in zip([tab1, tab2, tab3], chart_map.items()):
    with tab:
        path = Path(path_str)
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.warning("Run main.py first to generate figures.")

st.markdown("---")

# ---------------------------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------------------------

with st.expander("Methodology — Headcount Formula"):
    st.markdown("""
**Core formula (derived from Erlang-C, simplified for monthly planning):**

```
Required Hours = Contact Volume × Blended AHT (min) / 60
Gross Hours    = Required Hours / Occupancy Rate
Headcount      = ⌈ Gross Hours / Productive Hours per Agent ⌉
```

The ceiling function (⌈⌉) is applied because headcount must be a whole number,
and you always round up — under-staffing is more costly than a marginal over-hire.

**Blended AHT** is a weighted average across channels at the end-of-period mix:
- Phone: 8 min × 38% share = 3.04
- Chat:  6 min × 40% share = 2.40
- Email: 12 min × 22% share = 2.64
- **Blended: 8.1 minutes**

**Productive hours per agent per month:**
22 working days × 8 hours × (1 − 25% shrinkage) = **132 hours**

Shrinkage covers annual leave, sick leave, training, team meetings, and comfort breaks.

**Scenarios:**
- **Base** — forecast as-is, 85% occupancy (standard operating)
- **Growth** — forecast + 20%, 85% occupancy (planned business expansion)
- **Stress** — forecast + 40%, 80% occupancy (peak with degraded efficiency)

The stress scenario is not a prediction. It is a risk quantification: if growth
assumptions are wrong by 40%, this is the headcount exposure.
    """)
