"""
variance_analysis.py
--------------------
Budget vs actual cost variance decomposition for ecommerce-cx-analytics.

The total variance between budgeted and actual contact centre costs
is decomposed into three independent components:

  1. Volume variance  — caused by getting more or fewer contacts than budgeted
  2. Rate variance    — caused by cost per hour changing vs the budgeted rate
  3. Mix variance     — caused by the channel mix shifting vs budget assumptions
                        (more phone = higher average cost even if rates held)

This decomposition is standard management accounting practice. It tells
leadership not just *how much* the budget was missed by, but *why*.

Formula:
  Total Variance = Volume Variance + Rate Variance + Mix Variance

Where:
  Volume Variance = (Actual Hours - Budget Hours) × Budget Rate
  Rate Variance   = (Actual Rate  - Budget Rate)  × Actual Hours
  Mix Variance    = Total Variance - Volume Variance - Rate Variance

Outputs:
  - variance_summary.csv
  - Waterfall chart per brand + cross-brand summary waterfall
  - Monthly variance trend chart

Usage (CLI / GitHub):
    python src/budget/variance_analysis.py

Usage (Streamlit / import):
    from src.budget.variance_analysis import run_variance_analysis
    results = run_variance_analysis(costs_df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_COSTS  = 'data/raw/costs.csv'
FIGURES_DIR = 'outputs/figures'
OUTPUT_DIR  = 'outputs'

BRAND_COLORS = {
    'Takealot': '#1f77b4',
    'MrD':      '#ff7f0e',
    'Sellers':  '#2ca02c',
    'TFS':      '#d62728',
}


# ---------------------------------------------------------------------------
# VARIANCE DECOMPOSITION
# ---------------------------------------------------------------------------

def decompose_variance(costs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose budget vs actual variance at the brand × channel × month level.

    Volume Variance:
      How much of the total variance is explained purely by contact volume
      being higher or lower than budgeted, holding the rate constant?
      = (Actual Hours - Budget Hours) × Budget Rate

    Rate Variance:
      How much is explained by cost per hour moving vs budget, holding
      actual volume constant?
      = (Actual Rate - Budget Rate) × Actual Hours

    Mix Variance:
      The residual — what is not explained by volume or rate alone.
      Captures channel mix shift: if more contacts came through phone than
      budgeted, average cost rises even if individual channel rates held.
      = Total Variance - Volume Variance - Rate Variance
    """
    df = costs_df.copy()

    df['actual_hours'] = df['agent_hours']
    df['budget_hours'] = (df['budget_cost'] / df['cost_per_hour']).round(2)

    # Budget rate back-calculated from budget cost and budget hours
    df['budget_rate'] = np.where(
        df['budget_hours'] > 0,
        df['budget_cost'] / df['budget_hours'],
        df['cost_per_hour']
    )

    df['total_variance']  = (df['actual_cost'] - df['budget_cost']).round(2)
    df['volume_variance'] = ((df['actual_hours'] - df['budget_hours']) * df['budget_rate']).round(2)
    df['rate_variance']   = ((df['cost_per_hour'] - df['budget_rate']) * df['actual_hours']).round(2)
    df['mix_variance']    = (df['total_variance'] - df['volume_variance'] - df['rate_variance']).round(2)
    df['variance_flag']   = np.where(df['total_variance'] > 0, 'Over', 'Under')

    return df


def summarise_by_brand(variance_df: pd.DataFrame) -> pd.DataFrame:
    """Annual brand-level summary of all variance components."""
    summary = (
        variance_df
        .groupby('brand')[['actual_cost', 'budget_cost', 'total_variance',
                           'volume_variance', 'rate_variance', 'mix_variance']]
        .sum()
        .round(2)
        .reset_index()
    )
    summary['variance_pct'] = (
        summary['total_variance'] / summary['budget_cost'] * 100
    ).round(1)
    return summary


# ---------------------------------------------------------------------------
# WATERFALL CHART
# ---------------------------------------------------------------------------

def plot_waterfall(
    budget: float,
    volume_var: float,
    rate_var: float,
    mix_var: float,
    actual: float,
    brand: str,
    figures_dir: str,
) -> None:
    """
    Waterfall chart for a single brand.
    Bars float from the running cumulative total to show how each
    variance component builds from budget to actual.
    """
    labels     = ['Budget', 'Volume\nVariance', 'Rate\nVariance', 'Mix\nVariance', 'Actual']
    values     = [budget, volume_var, rate_var, mix_var, actual]

    # Cumulative bottom positions for floating bars
    bottoms    = [0, budget, budget + volume_var, budget + volume_var + rate_var, 0]
    bar_colors = ['#1f4e79', '', '', '', '#1f4e79']
    for i, v in enumerate([None, volume_var, rate_var, mix_var, None]):
        if v is not None:
            bar_colors[i] = '#d62728' if v >= 0 else '#2ca02c'

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, value, bottom, color) in enumerate(
            zip(labels, values, bottoms, bar_colors)):
        ax.bar(i, value, bottom=bottom, color=color, alpha=0.88,
               edgecolor='white', linewidth=1.2, width=0.6)

        y_pos   = bottom + value / 2
        sign    = '+' if value > 0 else ''
        lbl_txt = f'R{value:,.0f}' if i in (0, 4) else f'{sign}R{value:,.0f}'
        ax.text(i, y_pos, lbl_txt, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    # Dashed connector lines
    running = budget
    for i in range(1, 4):
        ax.plot([i - 0.7 + 0.4, i - 0.3 + 0.3],
                [running, running],
                color='grey', linewidth=0.8, linestyle='--')
        running += values[i]

    variance_pct = (actual - budget) / budget * 100
    sign = '+' if variance_pct > 0 else ''

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Cost (ZAR)')
    ax.set_title(
        f'Budget Variance Waterfall — {brand}\n'
        f'Total variance: {sign}R{(actual - budget):,.0f}  ({sign}{variance_pct:.1f}%)',
        fontsize=12, fontweight='bold'
    )
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)

    legend_elements = [
        Patch(facecolor='#1f4e79', label='Budget / Actual'),
        Patch(facecolor='#d62728', label='Unfavourable (over budget)'),
        Patch(facecolor='#2ca02c', label='Favourable (under budget)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    plt.tight_layout()
    path = f'{figures_dir}/variance_waterfall_{brand.lower()}.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved waterfall: {path}')


def plot_cross_brand_waterfall(summary_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Grouped bar chart: volume, rate, and mix variance side by side
    for all four brands. Executive-level cross-brand view.
    """
    brands      = summary_df['brand'].tolist()
    x           = np.arange(len(brands))
    width       = 0.25
    components  = ['volume_variance', 'rate_variance', 'mix_variance']
    comp_labels = ['Volume', 'Rate', 'Mix']
    comp_colors = ['#1f77b4', '#ff7f0e', '#9467bd']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (col, label, color) in enumerate(zip(components, comp_labels, comp_colors)):
        values = summary_df[col].values
        bars   = ax.bar(x + (i - 1) * width, values, width,
                        label=f'{label} Variance', color=color,
                        alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, values):
            sign   = '+' if val >= 0 else ''
            offset = abs(val) * 0.03 * (1 if val >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + offset,
                    f'{sign}R{val/1e3:.0f}K',
                    ha='center', fontsize=8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(brands, fontsize=11)
    ax.set_ylabel('Variance (ZAR)')
    ax.set_title(
        'Cross-Brand Variance Decomposition — Annual\n'
        'Positive = over budget (unfavourable) | Negative = under budget (favourable)',
        fontsize=12, fontweight='bold'
    )
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R{x:,.0f}'))
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{figures_dir}/variance_cross_brand.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved cross-brand variance: {path}')


def plot_monthly_variance_trend(variance_df: pd.DataFrame, figures_dir: str) -> None:
    """
    Line chart: total monthly variance per brand over 24 months.
    Orange shading marks Black Friday months where variance spikes.
    """
    monthly = (
        variance_df
        .groupby(['brand', 'month'])['total_variance']
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    for brand in sorted(monthly['brand'].unique()):
        data  = monthly[monthly['brand'] == brand].sort_values('month')
        color = BRAND_COLORS.get(brand, '#333')
        ax.plot(data['month'], data['total_variance'],
                color=color, linewidth=2, label=brand,
                marker='o', markersize=3)

    # Shade Black Friday months
    bf_months = monthly[monthly['month'].dt.month == 11]['month'].unique()
    for bf in bf_months:
        ax.axvspan(
            pd.Timestamp(bf),
            pd.Timestamp(bf) + pd.offsets.MonthEnd(1),
            alpha=0.12, color='orange'
        )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(
        'Monthly Budget Variance by Brand\n'
        'Orange shading = Black Friday | Above zero = over budget',
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel('Total Variance (ZAR)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R{x:,.0f}'))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{figures_dir}/variance_monthly_trend.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved monthly trend: {path}')


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_variance_analysis(costs_df: pd.DataFrame = None) -> dict:
    """
    Run the full variance analysis pipeline.

    Parameters
    ----------
    costs_df : pd.DataFrame, optional
        Pass in directly for Streamlit. If None, loads from disk (CLI).

    Returns
    -------
    dict with keys:
        'variance_detail' : full brand × channel × month decomposition
        'brand_summary'   : annual brand-level summary
    """
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if costs_df is None:
        costs_df = pd.read_csv(DATA_COSTS, parse_dates=['month'])
        print(f'Loaded costs  shape={costs_df.shape}')

    print('\nDecomposing variance into volume, rate, and mix components...')
    variance_df = decompose_variance(costs_df)
    summary_df  = summarise_by_brand(variance_df)

    out_path = f'{OUTPUT_DIR}/variance_summary.csv'
    summary_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')

    print('\nGenerating waterfall charts...')
    for _, row in summary_df.iterrows():
        plot_waterfall(
            budget      = row['budget_cost'],
            volume_var  = row['volume_variance'],
            rate_var    = row['rate_variance'],
            mix_var     = row['mix_variance'],
            actual      = row['actual_cost'],
            brand       = row['brand'],
            figures_dir = FIGURES_DIR,
        )

    plot_cross_brand_waterfall(summary_df, FIGURES_DIR)
    plot_monthly_variance_trend(variance_df, FIGURES_DIR)

    # --- Print summary ---
    print('\n--- Variance Summary (Annual, by Brand) ---')
    print(f'{"Brand":<12} {"Budget":>14} {"Actual":>14} {"Total Var":>12} '
          f'{"Vol Var":>12} {"Rate Var":>12} {"Mix Var":>12} {"% Over":>8}')
    print('-' * 100)
    for _, row in summary_df.iterrows():
        print(
            f'{row["brand"]:<12} '
            f'R{row["budget_cost"]:>12,.0f} '
            f'R{row["actual_cost"]:>12,.0f} '
            f'R{row["total_variance"]:>10,.0f} '
            f'R{row["volume_variance"]:>10,.0f} '
            f'R{row["rate_variance"]:>10,.0f} '
            f'R{row["mix_variance"]:>10,.0f} '
            f'{row["variance_pct"]:>7.1f}%'
        )

    print('\nVariance analysis complete.')

    return {
        'variance_detail': variance_df,
        'brand_summary':   summary_df,
    }


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_variance_analysis()
