"""
capacity_planning.py
--------------------
Agent headcount planning for ecommerce-cx-analytics.

Links the Holt-Winters contact volume forecast to agent headcount
requirements using Average Handle Time (AHT) and occupancy assumptions.

The core formula is derived from Erlang-C principles simplified for
monthly planning (as opposed to intra-day scheduling):

    Required Hours = Contact Volume × AHT (minutes) / 60
    Gross Hours    = Required Hours / Occupancy Rate
    Headcount      = Gross Hours / Productive Hours Per Agent Per Month

Three scenarios are modelled:
  - Base:   forecast as-is, 85% occupancy
  - Growth: forecast + 20%, 85% occupancy (reflects business expansion)
  - Stress: forecast + 40%, 80% occupancy (peak period with degraded efficiency)

The growth and stress scenarios answer the question executives always ask:
"What happens to our cost base if the business grows faster than expected?"

Outputs:
  - capacity_plan.csv: full headcount plan saved to outputs/
  - Scenario comparison chart per brand
  - Cross-brand headcount summary chart

Usage (CLI / GitHub):
    python src/budget/capacity_planning.py

Usage (Streamlit / import):
    from src.budget.capacity_planning import run_capacity_planning
    results = run_capacity_planning(forecast_df)
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

FORECAST_PATH = 'outputs/forecast_summary.csv'
FIGURES_DIR   = 'outputs/figures'
OUTPUT_DIR    = 'outputs'

# Average Handle Time in minutes per contact (including after-contact work)
AHT_MINUTES = {
    'phone': 8,
    'chat':  6,
    'email': 12,
}

# Blended AHT (weighted average across channels)
# Weights approximate the channel mix at month 24 of the simulation
BLENDED_AHT_MINUTES = (
    AHT_MINUTES['phone'] * 0.38 +
    AHT_MINUTES['chat']  * 0.40 +
    AHT_MINUTES['email'] * 0.22
)

# Productive hours an agent works per month
# 22 working days × 8 hours × shrinkage adjustment (leave, training, breaks)
WORKING_DAYS_PER_MONTH  = 22
HOURS_PER_DAY           = 8
SHRINKAGE               = 0.25   # 25% of time lost to non-productive activity
PRODUCTIVE_HOURS        = WORKING_DAYS_PER_MONTH * HOURS_PER_DAY * (1 - SHRINKAGE)

# Scenario definitions
SCENARIOS = {
    'Base':   {'volume_multiplier': 1.00, 'occupancy': 0.85},
    'Growth': {'volume_multiplier': 1.20, 'occupancy': 0.85},
    'Stress': {'volume_multiplier': 1.40, 'occupancy': 0.80},
}

SCENARIO_COLORS = {
    'Base':   '#1f77b4',
    'Growth': '#ff7f0e',
    'Stress': '#d62728',
}

BRAND_COLORS = {
    'Takealot': '#1f77b4',
    'MrD':      '#ff7f0e',
    'Sellers':  '#2ca02c',
    'TFS':      '#d62728',
}

# Monthly cost per agent FTE (ZAR, fully loaded: salary + benefits + overheads)
COST_PER_AGENT_MONTHLY = 28_000


# ---------------------------------------------------------------------------
# HEADCOUNT CALCULATION
# ---------------------------------------------------------------------------

def calculate_headcount(
    contact_volume: float,
    aht_minutes: float,
    occupancy: float,
    productive_hours: float,
) -> dict:
    """
    Calculate required agent headcount for a given contact volume.

    Steps:
      1. Convert contacts to required agent-hours (volume × AHT / 60)
      2. Gross up for occupancy (agents are not handling contacts 100% of the time)
      3. Divide by productive hours per agent per month to get headcount
      4. Round up — you cannot employ 0.7 of an agent

    Parameters
    ----------
    contact_volume  : total contacts in the period
    aht_minutes     : average handle time per contact in minutes
    occupancy       : proportion of logged-in time spent handling contacts
    productive_hours: productive hours per agent per month

    Returns
    -------
    dict with required_hours, gross_hours, headcount, monthly_cost
    """
    required_hours = (contact_volume * aht_minutes) / 60
    gross_hours    = required_hours / occupancy
    headcount      = int(np.ceil(gross_hours / productive_hours))
    monthly_cost   = headcount * COST_PER_AGENT_MONTHLY

    return {
        'required_hours':  round(required_hours, 1),
        'gross_hours':     round(gross_hours, 1),
        'headcount':       headcount,
        'monthly_cost':    monthly_cost,
    }


def build_capacity_plan(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full capacity plan across all brands, months, and scenarios.

    forecast_df must have columns: Brand, and one column per forecast month.
    Reshapes to long format, applies headcount calculation per scenario.

    Returns one row per brand × month × scenario.
    """
    # Reshape forecast from wide to long
    # Expected columns: Brand, MAPE (%), Jan 2024 Fcst, Feb 2024 Fcst, ...
    month_cols = [c for c in forecast_df.columns if 'Fcst' in c]

    long = forecast_df.melt(
        id_vars=['Brand'],
        value_vars=month_cols,
        var_name='month_label',
        value_name='forecast_volume'
    )

    # Parse month label to datetime
    long['month'] = pd.to_datetime(
        long['month_label'].str.replace(' Fcst', '', regex=False),
        format='%b %Y'
    )
    long = long.drop(columns=['month_label']).rename(columns={'Brand': 'brand'})

    rows = []
    for _, row in long.iterrows():
        for scenario, params in SCENARIOS.items():
            adj_volume = row['forecast_volume'] * params['volume_multiplier']
            hc = calculate_headcount(
                contact_volume  = adj_volume,
                aht_minutes     = BLENDED_AHT_MINUTES,
                occupancy       = params['occupancy'],
                productive_hours = PRODUCTIVE_HOURS,
            )
            rows.append({
                'brand':            row['brand'],
                'month':            row['month'],
                'scenario':         scenario,
                'forecast_volume':  row['forecast_volume'],
                'adjusted_volume':  round(adj_volume, 0),
                'volume_multiplier': params['volume_multiplier'],
                'occupancy':        params['occupancy'],
                **hc,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------------------------

def plot_scenario_comparison_by_brand(capacity_df: pd.DataFrame, figures_dir: str) -> None:
    """
    One 2×2 figure with a subplot per brand.
    Each subplot shows headcount under all three scenarios across 6 months.
    """
    brands = sorted(capacity_df['brand'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for i, brand in enumerate(brands):
        ax   = axes[i]
        data = capacity_df[capacity_df['brand'] == brand]

        for scenario in ['Base', 'Growth', 'Stress']:
            s_data = data[data['scenario'] == scenario].sort_values('month')
            ax.plot(
                s_data['month'], s_data['headcount'],
                color=SCENARIO_COLORS[scenario], linewidth=2.2,
                marker='o', markersize=5, label=scenario
            )
            # Annotate last point
            last = s_data.iloc[-1]
            ax.annotate(
                f'{last["headcount"]}',
                xy=(last['month'], last['headcount']),
                xytext=(5, 3), textcoords='offset points',
                fontsize=8, color=SCENARIO_COLORS[scenario], fontweight='bold'
            )

        ax.set_title(brand, fontsize=12, fontweight='bold')
        ax.set_ylabel('Required Agents (FTE)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Capacity Plan — Required Headcount by Brand and Scenario\n'
        f'Blended AHT = {BLENDED_AHT_MINUTES:.1f} min | '
        f'Productive hrs/agent/month = {PRODUCTIVE_HOURS:.0f} | '
        f'Shrinkage = {SHRINKAGE:.0%}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = f'{figures_dir}/capacity_scenario_by_brand.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_total_headcount_scenarios(capacity_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Stacked bar chart: total headcount across all brands per scenario.
    Shows the executive view — how many agents does the whole operation need?
    """
    total = (
        capacity_df
        .groupby(['month', 'scenario'])['headcount']
        .sum()
        .reset_index()
    )

    scenarios = ['Base', 'Growth', 'Stress']
    months    = sorted(total['month'].unique())
    x         = np.arange(len(months))
    width     = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, scenario in enumerate(scenarios):
        s_data = total[total['scenario'] == scenario].sort_values('month')
        bars   = ax.bar(
            x + (i - 1) * width,
            s_data['headcount'],
            width,
            label=scenario,
            color=SCENARIO_COLORS[scenario],
            alpha=0.85,
            edgecolor='white'
        )
        for bar, val in zip(bars, s_data['headcount']):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val),
                ha='center', fontsize=7.5, fontweight='bold'
            )

    month_labels = [m.strftime('%b %Y') for m in months]
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=45, fontsize=9)
    ax.set_ylabel('Total Required Agents (FTE)')
    ax.set_title(
        'Total Contact Centre Headcount Requirement — All Brands\n'
        'Base | Growth (+20%) | Stress (+40%)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{figures_dir}/capacity_total_headcount.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_monthly_cost_scenarios(capacity_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Line chart: total monthly agent cost (ZAR) per scenario across 6 months.
    Translates headcount into rand cost for the budget conversation.
    """
    total_cost = (
        capacity_df
        .groupby(['month', 'scenario'])['monthly_cost']
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    for scenario in ['Base', 'Growth', 'Stress']:
        data = total_cost[total_cost['scenario'] == scenario].sort_values('month')
        ax.plot(
            data['month'], data['monthly_cost'],
            color=SCENARIO_COLORS[scenario], linewidth=2.5,
            marker='o', markersize=6, label=scenario
        )

    ax.set_title(
        'Monthly Agent Cost by Scenario — All Brands Combined\n'
        f'Cost per FTE = R{COST_PER_AGENT_MONTHLY:,.0f}/month (fully loaded)',
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel('Monthly Agent Cost (ZAR)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R{x:,.0f}'))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{figures_dir}/capacity_monthly_cost.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_capacity_planning(forecast_df: pd.DataFrame = None) -> dict:
    """
    Run the full capacity planning pipeline.

    Parameters
    ----------
    forecast_df : pd.DataFrame, optional
        Forecast summary with Brand + month forecast columns.
        If None, loads from outputs/forecast_summary.csv (CLI use).

    Returns
    -------
    dict with keys:
        'capacity_plan'  : full brand × month × scenario DataFrame
        'summary'        : peak headcount and cost per brand per scenario
    """
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if forecast_df is None:
        forecast_df = pd.read_csv(FORECAST_PATH)
        print(f'Loaded forecast summary  shape={forecast_df.shape}')

    print(f'\nAssumptions:')
    print(f'  Blended AHT          : {BLENDED_AHT_MINUTES:.1f} minutes')
    print(f'  Productive hrs/agent : {PRODUCTIVE_HOURS:.0f} hours/month')
    print(f'  Shrinkage            : {SHRINKAGE:.0%}')
    print(f'  Cost per FTE         : R{COST_PER_AGENT_MONTHLY:,.0f}/month')
    print(f'  Scenarios            : {list(SCENARIOS.keys())}')

    print('\nBuilding capacity plan...')
    capacity_df = build_capacity_plan(forecast_df)

    # Save
    out_path = f'{OUTPUT_DIR}/capacity_plan.csv'
    capacity_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')

    # Figures
    print('\nGenerating figures...')
    plot_scenario_comparison_by_brand(capacity_df, FIGURES_DIR)
    plot_total_headcount_scenarios(capacity_df, FIGURES_DIR)
    plot_monthly_cost_scenarios(capacity_df, FIGURES_DIR)

    # Summary: peak month headcount and cost per brand per scenario
    summary = (
        capacity_df
        .groupby(['brand', 'scenario'])[['headcount', 'monthly_cost']]
        .max()
        .reset_index()
        .rename(columns={'headcount': 'peak_headcount', 'monthly_cost': 'peak_monthly_cost'})
    )

    print('\n--- Capacity Plan Summary (Peak Month per Brand per Scenario) ---')
    print(f'{"Brand":<12} {"Scenario":<10} {"Peak HC":>10} {"Peak Cost (ZAR)":>18}')
    print('-' * 54)
    for _, row in summary.sort_values(['brand', 'scenario']).iterrows():
        print(
            f'{row["brand"]:<12} '
            f'{row["scenario"]:<10} '
            f'{row["peak_headcount"]:>10} '
            f'R{row["peak_monthly_cost"]:>15,.0f}'
        )

    print('\nCapacity planning complete.')

    return {
        'capacity_plan': capacity_df,
        'summary':       summary,
    }


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_capacity_planning()
