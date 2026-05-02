"""
main.py
-------
End-to-end pipeline runner for ecommerce-cx-analytics.

Runs all five modules in sequence:
  1. Data generation
  2. Forecasting (seasonal decomposition + Holt-Winters)
  3. Cost allocation
  4. Budget variance analysis
  5. Executive dashboard

Usage:
    python main.py
    python main.py --skip-data   # skip data generation if CSVs already exist
"""

import argparse
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(skip_data: bool = False):
    start = time.time()
    print("=" * 60)
    print("  eCommerce CX Analytics — Full Pipeline")
    print("=" * 60)

    # --- Step 1: Data Generation ---
    if skip_data and Path('data/raw/contacts.csv').exists():
        print("\n[1/5] Data generation — SKIPPED (CSVs already exist)")
        import pandas as pd
        datasets = {
            'contacts':     pd.read_csv('data/raw/contacts.csv',     parse_dates=['month']),
            'orders':       pd.read_csv('data/raw/orders.csv',       parse_dates=['month']),
            'costs':        pd.read_csv('data/raw/costs.csv',        parse_dates=['month']),
            'satisfaction': pd.read_csv('data/raw/satisfaction.csv', parse_dates=['month']),
        }
    else:
        print("\n[1/5] Generating synthetic data...")
        from src.data_generation.generate_data import generate_all_data
        datasets = generate_all_data(output_dir='data/raw', save=True)

    # --- Step 2: Forecasting ---
    print("\n[2/5] Running contact volume forecasting...")
    from src.forecasting.time_series import run_forecasting
    forecast_results = run_forecasting(contacts_df=datasets['contacts'])

    # --- Step 3: Cost Allocation ---
    print("\n[3/5] Running cost allocation...")
    from src.budget.cost_allocation import run_cost_allocation
    allocation_results = run_cost_allocation(
        contacts_df=datasets['contacts'],
        costs_df=datasets['costs'],
    )

    # --- Step 4: Variance Analysis ---
    print("\n[4/5] Running budget variance analysis...")
    from src.budget.variance_analysis import run_variance_analysis
    variance_results = run_variance_analysis(costs_df=datasets['costs'])

    # --- Step 5: Executive Dashboard ---
    print("\n[5/5] Building executive dashboard...")
    from src.reporting.executive_dashboard import run_dashboard
    run_dashboard(
        contacts_df     = datasets['contacts'],
        costs_df        = datasets['costs'],
        satisfaction_df = datasets['satisfaction'],
        forecast_df     = forecast_results['summary'],
        variance_df     = variance_results['brand_summary'],
    )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Dashboard : outputs/executive_dashboard.png")
    print(f"  Narrative : outputs/reports/executive_summary.txt")
    print(f"  Figures   : outputs/figures/ ({len(list(Path('outputs/figures').glob('*.png')))} files)")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eCommerce CX Analytics Pipeline')
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data generation if raw CSVs already exist'
    )
    args = parser.parse_args()
    run_pipeline(skip_data=args.skip_data)
