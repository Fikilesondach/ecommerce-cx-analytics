"""
cost_allocation.py
------------------
Shared service cost distribution across brands for ecommerce-cx-analytics.

Shared services (contact centre management, QA, telephony infrastructure,
workforce management) cannot be attributed to a single brand. This module
distributes them using two allocation keys:

  1. Contact volume weight  — brand's share of total monthly contacts
  2. Channel-adjusted weight — volume weight modified by channel mix,
     because phone contacts consume more shared infrastructure than chat

The channel-adjusted method is more defensible in an executive review
because it reflects actual resource consumption, not just headcount.

Outputs:
  - allocated_costs.csv: full allocation table saved to outputs/
  - Three figures: allocation waterfall, brand share trend, method comparison

Usage (CLI / GitHub):
    python src/budget/cost_allocation.py

Usage (Streamlit / import):
    from src.budget.cost_allocation import run_cost_allocation
    results = run_cost_allocation(contacts_df, costs_df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_CONTACTS = 'data/raw/contacts.csv'
DATA_COSTS    = 'data/raw/costs.csv'
FIGURES_DIR   = 'outputs/figures'
OUTPUT_DIR    = 'outputs'

BRAND_COLORS = {
    'Takealot': '#1f77b4',
    'MrD':      '#ff7f0e',
    'Sellers':  '#2ca02c',
    'TFS':      '#d62728',
}

# Channel cost weights — phone consumes more shared infrastructure
# These weights adjust the raw volume allocation
CHANNEL_COST_WEIGHT = {
    'phone': 1.5,   # 50% more resource-intensive than baseline
    'chat':  1.0,   # baseline
    'email': 0.7,   # less real-time infrastructure required
}


# ---------------------------------------------------------------------------
# ALLOCATION ENGINE
# ---------------------------------------------------------------------------

def compute_volume_weights(contacts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Method 1 — Simple volume weighting.

    Each brand receives a share of shared costs equal to its share
    of total contact volume in that month.

    brand_share(month) = brand_contacts(month) / total_contacts(month)
    """
    monthly_brand = (
        contacts_df
        .groupby(['brand', 'month'])['contact_volume']
        .sum()
        .reset_index()
    )

    monthly_total = (
        contacts_df
        .groupby('month')['contact_volume']
        .sum()
        .reset_index()
        .rename(columns={'contact_volume': 'total_volume'})
    )

    merged = monthly_brand.merge(monthly_total, on='month')
    merged['volume_weight'] = merged['contact_volume'] / merged['total_volume']

    return merged[['brand', 'month', 'contact_volume', 'total_volume', 'volume_weight']]


def compute_channel_adjusted_weights(contacts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Method 2 — Channel-adjusted weighting.

    Weighted contact volume = sum over channels of (volume × channel_cost_weight).
    Brands with a higher phone mix absorb more shared costs because phone
    contacts require more real-time infrastructure (telephony, workforce mgmt).

    This is the more defensible method for executive review.
    """
    df = contacts_df.copy()
    df['weighted_volume'] = df['contact_volume'] * df['channel'].map(CHANNEL_COST_WEIGHT)

    monthly_brand_weighted = (
        df
        .groupby(['brand', 'month'])['weighted_volume']
        .sum()
        .reset_index()
    )

    monthly_total_weighted = (
        df
        .groupby('month')['weighted_volume']
        .sum()
        .reset_index()
        .rename(columns={'weighted_volume': 'total_weighted_volume'})
    )

    merged = monthly_brand_weighted.merge(monthly_total_weighted, on='month')
    merged['channel_adjusted_weight'] = (
        merged['weighted_volume'] / merged['total_weighted_volume']
    )

    return merged[['brand', 'month', 'weighted_volume',
                   'total_weighted_volume', 'channel_adjusted_weight']]


def allocate_shared_costs(
    contacts_df: pd.DataFrame,
    costs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply both allocation methods to the shared service costs in costs_df.

    Shared service allocation in costs_df is a per-brand monthly overhead.
    We first sum it to a total monthly shared pool, then redistribute it
    using both weighting methods so they can be compared.

    Returns a DataFrame with one row per brand per month containing:
      - raw shared_service_allocation (as recorded in costs.csv)
      - volume_allocated_cost    (Method 1)
      - channel_allocated_cost   (Method 2)
      - difference between methods (shows impact of channel adjustment)
    """
    # Total shared service pool per month (sum across all brands and channels)
    monthly_shared_pool = (
        costs_df
        .groupby('month')['shared_service_allocation']
        .sum()
        .reset_index()
        .rename(columns={'shared_service_allocation': 'shared_pool'})
    )

    # Compute both weight sets
    vol_weights  = compute_volume_weights(contacts_df)
    chan_weights = compute_channel_adjusted_weights(contacts_df)

    # Merge weights with the shared pool
    allocation = vol_weights.merge(chan_weights[['brand', 'month', 'channel_adjusted_weight']],
                                   on=['brand', 'month'])
    allocation = allocation.merge(monthly_shared_pool, on='month')

    # Apply allocation
    allocation['volume_allocated_cost'] = (
        allocation['volume_weight'] * allocation['shared_pool']
    ).round(2)

    allocation['channel_allocated_cost'] = (
        allocation['channel_adjusted_weight'] * allocation['shared_pool']
    ).round(2)

    allocation['allocation_difference'] = (
        allocation['channel_allocated_cost'] - allocation['volume_allocated_cost']
    ).round(2)

    return allocation


# ---------------------------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------------------------

def plot_brand_share_trend(allocation_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Line chart: each brand's share of shared costs over 24 months.
    Both allocation methods shown as solid vs dashed lines.
    """
    brands = sorted(allocation_df['brand'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for i, brand in enumerate(brands):
        ax   = axes[i]
        data = allocation_df[allocation_df['brand'] == brand].sort_values('month')
        color = BRAND_COLORS.get(brand, '#333')

        ax.plot(data['month'], data['volume_weight'] * 100,
                color=color, linewidth=2, label='Volume weight')
        ax.plot(data['month'], data['channel_adjusted_weight'] * 100,
                color=color, linewidth=2, linestyle='--',
                label='Channel-adjusted weight')

        ax.set_title(brand, fontsize=12, fontweight='bold')
        ax.set_ylabel('Share of Shared Costs (%)')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Shared Cost Allocation — Brand Share Over Time\n'
        'Solid = volume weight | Dashed = channel-adjusted weight',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    path = f'{figures_dir}/allocation_brand_share_trend.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_annual_allocation_comparison(allocation_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Grouped bar chart: annual total allocated cost per brand,
    comparing volume method vs channel-adjusted method.
    Shows which brands win and lose under the more nuanced method.
    """
    annual = (
        allocation_df
        .groupby('brand')[['volume_allocated_cost', 'channel_allocated_cost']]
        .sum()
        .reset_index()
    )

    brands = annual['brand'].tolist()
    x      = np.arange(len(brands))
    width  = 0.35
    colors = [BRAND_COLORS.get(b, '#333') for b in brands]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, annual['volume_allocated_cost'],
                   width, label='Volume weight', alpha=0.6,
                   color=colors, edgecolor='white')
    bars2 = ax.bar(x + width/2, annual['channel_allocated_cost'],
                   width, label='Channel-adjusted', alpha=1.0,
                   color=colors, edgecolor='white')

    # Difference labels above channel-adjusted bars
    for i, row in annual.iterrows():
        diff = row['channel_allocated_cost'] - row['volume_allocated_cost']
        sign = '+' if diff >= 0 else ''
        ax.text(
            i + width/2,
            row['channel_allocated_cost'] + 5000,
            f'{sign}R{diff:,.0f}',
            ha='center', fontsize=9,
            color='green' if diff >= 0 else 'red'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(brands, fontsize=11)
    ax.set_ylabel('Annual Allocated Shared Cost (ZAR)')
    ax.set_title(
        'Shared Cost Allocation — Method Comparison (Annual Total)\n'
        'Difference shows impact of adjusting for channel mix',
        fontsize=12, fontweight='bold'
    )
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R{x:,.0f}'))
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{figures_dir}/allocation_method_comparison.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_monthly_shared_pool(allocation_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Stacked area chart: monthly shared cost pool split by brand
    using the channel-adjusted allocation method.
    """
    pivot = (
        allocation_df
        .groupby(['month', 'brand'])['channel_allocated_cost']
        .sum()
        .unstack('brand')
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    brands  = pivot.columns.tolist()
    colors  = [BRAND_COLORS.get(b, '#333') for b in brands]

    ax.stackplot(pivot.index, pivot.T.values, labels=brands, colors=colors, alpha=0.8)

    ax.set_title(
        'Monthly Shared Cost Pool — Channel-Adjusted Allocation by Brand',
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel('Allocated Shared Cost (ZAR)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R{x:,.0f}'))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{figures_dir}/allocation_monthly_pool.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_cost_allocation(
    contacts_df: pd.DataFrame = None,
    costs_df: pd.DataFrame = None,
) -> dict:
    """
    Run the full cost allocation pipeline.

    Parameters
    ----------
    contacts_df, costs_df : pd.DataFrame, optional
        Pass in directly for Streamlit. If None, loads from disk (CLI).

    Returns
    -------
    dict with keys:
        'allocation'   : full allocation DataFrame (brand × month)
        'annual_summary': annual totals per brand, both methods
    """
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if contacts_df is None:
        contacts_df = pd.read_csv(DATA_CONTACTS, parse_dates=['month'])
        print(f'Loaded contacts  shape={contacts_df.shape}')
    if costs_df is None:
        costs_df = pd.read_csv(DATA_COSTS, parse_dates=['month'])
        print(f'Loaded costs     shape={costs_df.shape}')

    # --- Compute allocation ---
    print('\nComputing shared cost allocation...')
    allocation_df = allocate_shared_costs(contacts_df, costs_df)

    # --- Annual summary ---
    annual_summary = (
        allocation_df
        .groupby('brand')[['volume_allocated_cost', 'channel_allocated_cost',
                           'allocation_difference']]
        .sum()
        .round(2)
        .reset_index()
    )
    annual_summary['pct_difference'] = (
        annual_summary['allocation_difference']
        / annual_summary['volume_allocated_cost'] * 100
    ).round(1)

    # --- Save output ---
    out_path = f'{OUTPUT_DIR}/allocated_costs.csv'
    allocation_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')

    # --- Figures ---
    print('\nGenerating figures...')
    plot_brand_share_trend(allocation_df, FIGURES_DIR)
    plot_annual_allocation_comparison(allocation_df, FIGURES_DIR)
    plot_monthly_shared_pool(allocation_df, FIGURES_DIR)

    # --- Print summary ---
    print('\n--- Annual Allocation Summary ---')
    print(f'{"Brand":<12} {"Volume Method":>15} {"Channel Method":>16} {"Difference":>12} {"% Diff":>8}')
    print('-' * 65)
    for _, row in annual_summary.iterrows():
        sign = '+' if row['allocation_difference'] >= 0 else ''
        print(
            f'{row["brand"]:<12} '
            f'R{row["volume_allocated_cost"]:>13,.0f} '
            f'R{row["channel_allocated_cost"]:>14,.0f} '
            f'{sign}R{row["allocation_difference"]:>10,.0f} '
            f'{sign}{row["pct_difference"]:>6.1f}%'
        )

    print('\nCost allocation complete.')

    return {
        'allocation':     allocation_df,
        'annual_summary': annual_summary,
    }


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_cost_allocation()
