"""
3_cost_allocation.py
--------------------
Streamlit page: Shared Service Cost Allocation
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.generate_data import generate_all_data
from src.budget.cost_allocation import run_cost_allocation

st.set_page_config(page_title="Cost Allocation", page_icon="📦", layout="wide")
st.title("📦 Shared Service Cost Allocation")
st.caption("Distributing contact centre overheads across brands — volume vs channel-adjusted methods")
st.markdown("---")

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_data():
    return generate_all_data(save=False)

@st.cache_data(show_spinner="Running cost allocation...")
def load_allocation(contacts_df, costs_df):
    return run_cost_allocation(contacts_df=contacts_df, costs_df=costs_df)

datasets           = load_data()
allocation_results = load_allocation(datasets['contacts'], datasets['costs'])
annual             = allocation_results['annual_summary']

# ---------------------------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------------------------

total_pool = annual['volume_allocated_cost'].sum()
st.metric("Total Shared Service Pool (Annual)", f"R{total_pool/1e6:.2f}M",
          "Distributed across 4 brands")

st.markdown("---")

# ---------------------------------------------------------------------------
# METHOD COMPARISON TABLE
# ---------------------------------------------------------------------------

st.subheader("Annual Allocation by Brand — Both Methods")
st.caption("Channel-adjusted method weights phone contacts more heavily than chat or email.")

display = annual.copy()
display['volume_allocated_cost']  = display['volume_allocated_cost'].apply(lambda x: f"R{x:,.0f}")
display['channel_allocated_cost'] = display['channel_allocated_cost'].apply(lambda x: f"R{x:,.0f}")
display['allocation_difference']  = display['allocation_difference'].apply(
    lambda x: f"+R{x:,.0f}" if x >= 0 else f"R{x:,.0f}"
)
display['pct_difference'] = display['pct_difference'].apply(
    lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%"
)
display.columns = ['Brand', 'Volume Method', 'Channel-Adjusted', 'Difference', '% Diff']
st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------

st.subheader("Allocation Charts")

tab1, tab2, tab3 = st.tabs([
    "Brand Share Trend",
    "Method Comparison",
    "Monthly Pool (Stacked)"
])

chart_map = {
    "Brand Share Trend":        "outputs/figures/allocation_brand_share_trend.png",
    "Method Comparison":        "outputs/figures/allocation_method_comparison.png",
    "Monthly Pool (Stacked)":   "outputs/figures/allocation_monthly_pool.png",
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

with st.expander("Methodology — Allocation Methods"):
    st.markdown("""
**What are shared service costs?**

Shared services are contact centre costs that cannot be attributed to a single brand:
management team salaries, QA function, telephony infrastructure, workforce management
systems, training budgets. They must be allocated across brands using a fair key.

**Method 1 — Simple Volume Weight**

Each brand receives a share of shared costs equal to its share of total contact volume.

```
brand_share = brand_contacts / total_contacts
allocated_cost = brand_share × shared_pool
```

Simple, auditable, easy to explain. The limitation: it ignores the fact that a phone
contact consumes more infrastructure than a chat contact.

**Method 2 — Channel-Adjusted Weight**

Volume is weighted by channel intensity before computing shares:
- Phone: 1.5× (real-time telephony, longer handle time)
- Chat: 1.0× (baseline)
- Email: 0.7× (asynchronous, lower infrastructure demand)

```
weighted_volume = Σ (channel_volume × channel_weight)
brand_share = brand_weighted_volume / total_weighted_volume
```

More defensible in an executive review because it reflects actual resource consumption.
In this simulation the differences are small because all brands share a similar
channel mix. In a real business, a logistics brand (heavy phone) vs a food delivery
brand (heavy chat) would show materially different allocations under the two methods.
    """)
