"""
generate_data.py
----------------
Synthetic data generation for the ecommerce-cx-analytics project.
Simulates 24 months (Jan 2022 – Dec 2023) of contact centre data
across four brands: Takealot, MrD, Sellers, TFS.

Usage (CLI / GitHub):
    python generate_data.py

Usage (Streamlit / import):
    from generate_data import generate_all_data
    datasets = generate_all_data(save=False)   # returns dict of DataFrames
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BRANDS = ['Takealot', 'MrD', 'Sellers', 'TFS']
CHANNELS = ['phone', 'chat', 'email']
MONTHS = pd.date_range(start='2022-01-01', periods=24, freq='MS')
SEED = 42

# Monthly seasonal multipliers (index = month number 1–12).
# November and December are the primary spikes for SA e-commerce.
SEASONAL_PATTERN = {
    1: 0.85,   # January  – post-holiday quiet
    2: 0.88,   # February
    3: 0.92,   # March
    4: 0.95,   # April
    5: 0.93,   # May
    6: 0.90,   # June
    7: 0.92,   # July
    8: 0.95,   # August
    9: 0.98,   # September
    10: 1.05,  # October  – pre-Black Friday build-up
    11: 1.40,  # November – Black Friday
    12: 1.35,  # December – Christmas peak
}

# Promo months (outside Black Friday): Feb, Jun, Aug
PROMO_MONTHS = {2, 6, 8}

# Black Friday month
BLACK_FRIDAY_MONTH = 11

# Brand baseline monthly contact volumes (total across all channels)
BRAND_CONTACT_BASELINE = {
    'Takealot': 5_000,   # high volume, broad product mix
    'MrD':      3_500,   # food delivery, highest frequency
    'Sellers':  1_000,   # B2B, low volume, higher complexity
    'TFS':      2_000,   # logistics-focused contacts
}

# Brand baseline monthly order volumes
BRAND_ORDER_BASELINE = {
    'Takealot': 80_000,
    'MrD':     120_000,  # very high frequency food orders
    'Sellers':   5_000,
    'TFS':      15_000,
}

# Cost per hour by channel (ZAR). Phone is most expensive.
COST_PER_HOUR = {'phone': 280, 'chat': 200, 'email': 150}

# Average handle time in minutes (including after-contact work)
AHT_MINUTES = {'phone': 8, 'chat': 6, 'email': 12}

# Monthly shared service allocation per brand (ZAR)
# Covers management, QA, telephony infrastructure — allocated later by volume
SHARED_SERVICE_BASE = {
    'Takealot': 85_000,
    'MrD':      60_000,
    'Sellers':  25_000,
    'TFS':      40_000,
}

# Base CSAT (out of 10), NPS, and first-contact resolution by brand/channel
BASE_CSAT = {'Takealot': 7.8, 'MrD': 7.5, 'Sellers': 8.2, 'TFS': 7.0}
BASE_NPS  = {'Takealot': 42,  'MrD': 38,  'Sellers': 55,  'TFS': 30}
BASE_FCR  = {'phone': 0.82, 'chat': 0.75, 'email': 0.72}


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _channel_mix(month_idx: int) -> dict:
    """
    Returns the channel split for a given month index (0 = Jan 2022).

    Phone declines from 50% to ~35% as customers migrate to chat.
    Chat grows from 28% to ~43%.
    Email fills the remainder and stays relatively stable.
    """
    phone = max(0.50 - month_idx * 0.006, 0.35)
    chat  = min(0.28 + month_idx * 0.006, 0.43)
    email = round(1.0 - phone - chat, 4)
    return {'phone': phone, 'chat': chat, 'email': email}


def _yoy_growth(month_idx: int, annual_rate: float) -> float:
    """Linear approximation of year-over-year growth across 24 months."""
    return 1.0 + (month_idx / 24) * annual_rate


def _inflation_factor(month_idx: int, annual_rate: float = 0.06) -> float:
    """South African CPI-style cost inflation across the period."""
    return 1.0 + (month_idx / 24) * annual_rate


# ---------------------------------------------------------------------------
# DATASET GENERATORS
# ---------------------------------------------------------------------------

def _generate_contacts() -> pd.DataFrame:
    """
    contacts.csv — the spine of the project.

    One row per brand × channel × month (288 rows total).
    contact_volume is the primary target variable for forecasting.
    resolution_rate degrades during high-volume periods.
    """
    rows = []

    for month_idx, month in enumerate(MONTHS):
        season_mult   = SEASONAL_PATTERN[month.month]
        is_bf         = int(month.month == BLACK_FRIDAY_MONTH)
        mix           = _channel_mix(month_idx)
        growth        = _yoy_growth(month_idx, annual_rate=0.15)

        for brand in BRANDS:
            baseline = BRAND_CONTACT_BASELINE[brand]

            for channel in CHANNELS:
                channel_share = mix[channel]

                volume = int(
                    baseline
                    * growth
                    * season_mult
                    * channel_share
                    * np.random.uniform(0.95, 1.05)   # random noise ±5%
                )

                # Resolution degrades when volume spikes.
                # Base rate differs by channel: phone resolves most on first call.
                base_res = {'phone': 0.85, 'chat': 0.78, 'email': 0.82}
                resolution_rate = round(
                    base_res[channel]
                    - (season_mult - 1.0) * 0.15    # volume penalty
                    + np.random.uniform(-0.02, 0.02),
                    3
                )
                resolution_rate = float(np.clip(resolution_rate, 0.60, 0.95))

                rows.append({
                    'brand':            brand,
                    'channel':          channel,
                    'month':            month,
                    'contact_volume':   volume,
                    'resolution_rate':  resolution_rate,
                    'black_friday_flag': is_bf,
                })

    return pd.DataFrame(rows)


def _generate_orders() -> pd.DataFrame:
    """
    orders.csv — the primary contact driver.

    More orders → more contacts. Promotions and Black Friday
    create multiplicative spikes on top of the seasonal pattern.
    Returns volume generates a separate wave of contacts post-peak.
    """
    rows = []

    for month_idx, month in enumerate(MONTHS):
        season_mult = SEASONAL_PATTERN[month.month]
        is_bf       = int(month.month == BLACK_FRIDAY_MONTH)
        is_promo    = int(month.month in PROMO_MONTHS)
        growth      = _yoy_growth(month_idx, annual_rate=0.20)

        # Promotional and Black Friday multipliers on order volume
        promo_mult = 1.25 if is_promo else 1.0
        bf_mult    = 2.00 if is_bf    else 1.0

        for brand in BRANDS:
            baseline = BRAND_ORDER_BASELINE[brand]

            order_volume = int(
                baseline
                * growth
                * season_mult
                * promo_mult
                * bf_mult
                * np.random.uniform(0.97, 1.03)
            )

            # Returns: ~8–10% normally, ~12% in Black Friday aftermath
            return_rate   = 0.12 if is_bf else np.random.uniform(0.08, 0.10)
            returns_volume = int(order_volume * return_rate)

            rows.append({
                'brand':           brand,
                'month':           month,
                'order_volume':    order_volume,
                'returns_volume':  returns_volume,
                'promo_flag':      is_promo,
                'black_friday_flag': is_bf,
            })

    return pd.DataFrame(rows)


def _generate_costs(contacts_df: pd.DataFrame) -> pd.DataFrame:
    """
    costs.csv — what each contact costs, and what was budgeted.

    Agent hours are derived from contact volume × AHT.
    Actual cost = agent_hours × cost_per_hour (with CPI inflation).
    Budget cost is set at the start of each year with a deliberate
    underestimate, creating the variance the analysis module will explain.
    Shared service allocation is a fixed monthly overhead per brand,
    later redistributed by volume weight in the budget module.
    """
    rows = []

    # Build a lookup for fast contact volume retrieval
    contact_lookup = contacts_df.set_index(['brand', 'channel', 'month'])['contact_volume']

    for month_idx, month in enumerate(MONTHS):
        inflation = _inflation_factor(month_idx)

        for brand in BRANDS:
            for channel in CHANNELS:
                volume = contact_lookup.get((brand, channel, month), 0)

                aht        = AHT_MINUTES[channel]
                agent_hours = round((volume * aht) / 60, 1)

                cph         = COST_PER_HOUR[channel] * inflation
                actual_cost = round(agent_hours * cph, 2)

                # Budget: volume underestimated by 3–10%, rate underestimated by 0–5%
                budget_volume = volume * np.random.uniform(0.90, 0.97)
                budget_hours  = round((budget_volume * aht) / 60, 1)
                budget_cost   = round(
                    budget_hours * cph * np.random.uniform(0.95, 1.00), 2
                )

                shared_alloc = round(
                    SHARED_SERVICE_BASE[brand] * np.random.uniform(0.95, 1.05), 2
                )

                rows.append({
                    'brand':                    brand,
                    'channel':                  channel,
                    'month':                    month,
                    'agent_hours':              agent_hours,
                    'cost_per_hour':            round(cph, 2),
                    'actual_cost':              actual_cost,
                    'budget_cost':              budget_cost,
                    'shared_service_allocation': shared_alloc,
                })

    return pd.DataFrame(rows)


def _generate_satisfaction() -> pd.DataFrame:
    """
    satisfaction.csv — customer experience outcomes.

    CSAT and NPS are negatively correlated with contact volume spikes:
    when agents are overwhelmed in November/December, service degrades.
    First-contact resolution mirrors the resolution_rate pattern in contacts.csv
    but is computed independently to keep the datasets loosely coupled.
    """
    rows = []

    for month_idx, month in enumerate(MONTHS):
        season_mult  = SEASONAL_PATTERN[month.month]
        # How much worse does service get during peaks?
        sat_penalty  = (season_mult - 1.0) * 0.8

        for brand in BRANDS:
            for channel in CHANNELS:

                csat = round(
                    BASE_CSAT[brand]
                    - sat_penalty
                    + np.random.uniform(-0.2, 0.2),
                    2
                )

                nps = round(
                    BASE_NPS[brand]
                    - sat_penalty * 10
                    + np.random.uniform(-3, 3),
                    1
                )

                fcr = round(
                    BASE_FCR[channel]
                    - sat_penalty * 0.10
                    + np.random.uniform(-0.02, 0.02),
                    3
                )

                # Clip to realistic bounds
                csat = float(np.clip(csat, 4.0, 10.0))
                nps  = float(np.clip(nps,  -20,  80))
                fcr  = float(np.clip(fcr,   0.50, 0.95))

                rows.append({
                    'brand':                  brand,
                    'channel':                channel,
                    'month':                  month,
                    'csat_score':             csat,
                    'nps_score':              nps,
                    'first_contact_resolution': fcr,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def generate_all_data(output_dir: str = 'data/raw', save: bool = True) -> dict:
    """
    Generate all four synthetic datasets.

    Parameters
    ----------
    output_dir : str
        Directory to write CSV files. Created if it does not exist.
        Only used when save=True.
    save : bool
        If True, write CSVs to output_dir (GitHub / CLI use).
        If False, return DataFrames only (Streamlit / import use).

    Returns
    -------
    dict with keys: 'contacts', 'orders', 'costs', 'satisfaction'
    Each value is a pandas DataFrame.
    """
    np.random.seed(SEED)

    print("Generating contacts...")
    contacts_df     = _generate_contacts()

    print("Generating orders...")
    orders_df       = _generate_orders()

    print("Generating costs...")
    costs_df        = _generate_costs(contacts_df)

    print("Generating satisfaction scores...")
    satisfaction_df = _generate_satisfaction()

    datasets = {
        'contacts':     contacts_df,
        'orders':       orders_df,
        'costs':        costs_df,
        'satisfaction': satisfaction_df,
    }

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for name, df in datasets.items():
            path = f'{output_dir}/{name}.csv'
            df.to_csv(path, index=False)
            print(f"  Saved {path}  shape={df.shape}")

    # Summary
    print("\nDataset shapes:")
    for name, df in datasets.items():
        print(f"  {name:15s}: {df.shape[0]:>4} rows × {df.shape[1]} cols")

    return datasets


if __name__ == '__main__':
    generate_all_data(output_dir='data/raw', save=True)
