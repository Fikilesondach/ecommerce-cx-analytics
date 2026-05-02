# ecommerce-cx-analytics

A contact centre analytics and forecasting engine for a simulated multi-brand South African e-commerce group.

Built to demonstrate end-to-end analytical capability across forecasting, budget modelling, cost allocation, capacity planning, and executive reporting — the core competencies required for senior CX analytics roles in financial services and e-commerce.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecommerce-cx-analytics-2bwkfuiappjkmz7gm8uvfgo.streamlit.app)

> **Live app:** [https://ecommerce-cx-analytics-6cznqxwthg2yd4nblzqjqt.streamlit.app](https://ecommerce-cx-analytics-6cznqxwthg2yd4nblzqjqt.streamlit.app)

---

## What This Project Does

A large e-commerce group runs a shared contact centre across four brands. Every month, customers call, chat, or email. Leadership needs to know three things in advance and one thing after the fact:

| Question | Module |
|---|---|
| How many contacts are coming? | Forecasting |
| What will it cost, and are we on budget? | Budget & Variance Analysis |
| How many agents do we need? | Capacity Planning |
| How did we actually perform? | Executive Dashboard |

This project answers all four questions from a single pipeline.

---

## Project Structure

```
ecommerce-cx-analytics/
│
├── data/
│   ├── raw/                      # Synthetic datasets (generated)
│   └── processed/                # Feature-engineered outputs
│
├── src/
│   ├── data_generation/
│   │   └── generate_data.py      # Synthetic multi-brand contact data
│   ├── forecasting/
│   │   └── time_series.py        # Seasonal decomposition + Holt-Winters
│   ├── budget/
│   │   ├── cost_allocation.py    # Shared service cost distribution
│   │   ├── variance_analysis.py  # Budget vs actual decomposition
│   │   └── capacity_planning.py  # Agent headcount model
│   └── reporting/
│       └── executive_dashboard.py # Four-panel executive dashboard
│
├── streamlit/
│   ├── app.py                    # Streamlit entry point
│   └── pages/                    # One page per module
│
├── notebooks/eda/
│   └── 01_eda.py                 # Exploratory data analysis
│
├── outputs/
│   ├── figures/                  # All generated charts
│   ├── reports/                  # Executive summary narrative
│   └── executive_dashboard.png   # Main deliverable
│
├── main.py                       # End-to-end pipeline runner
└── requirements.txt
```

---

## The Data

Four synthetic datasets covering 24 months (January 2022 – December 2023) across four brands and three channels.

| Dataset | Rows | Key Fields |
|---|---|---|
| contacts.csv | 288 | brand, channel, month, contact_volume, resolution_rate |
| orders.csv | 96 | brand, month, order_volume, returns_volume, promo_flag |
| costs.csv | 288 | brand, channel, month, actual_cost, budget_cost, agent_hours |
| satisfaction.csv | 288 | brand, channel, month, csat_score, nps_score, first_contact_resolution |

**Brands simulated:**
- **Takealot.com** — high volume, broad product mix
- **Mr D** — food delivery, highest contact frequency
- **Sellers** — B2B, lower volume, higher complexity
- **TFS (Takealot Fulfilment Services)** — logistics-focused contacts

**Realistic patterns built into the data:**
- November/December seasonal spikes (~40% above baseline)
- Black Friday contact volume doubling
- Phone channel declining, chat growing across the 24-month window
- CSAT degrading during high-volume months
- Budget deliberately underestimated by 3–10% to create realistic variance

---

## The Five Modules

### Module 1 — Contact Volume Forecasting (`src/forecasting/time_series.py`)

**Approach:** Two-stage per brand.

Stage 1 applies `statsmodels.tsa.seasonal.seasonal_decompose` to separate the contact volume series into trend, seasonal, and residual components. This is a diagnostic — it confirms the seasonal pattern is stable before fitting a model.

Stage 2 fits `statsmodels.tsa.holtwinters.ExponentialSmoothing` (triple exponential smoothing) on the full 24-month series and forecasts 6 months ahead. Holt-Winters handles both trend and seasonality in a single model with three smoothing parameters: level, trend, and seasonal.

**Validation:** In-sample MAPE on the final 6 months of fitted values.

| Brand | MAPE |
|---|---|
| Takealot | 0.9% |
| MrD | 1.0% |
| TFS | 1.1% |
| Sellers | 0.9% |

> Note: MAPEs below 1% reflect in-sample validation on a short series. With additional historical data, an out-of-time split would be used for a more conservative accuracy estimate.

---

### Module 2 — Budget Variance Analysis (`src/budget/variance_analysis.py`)

**Approach:** Standard management accounting decomposition.

Total variance (actual − budget) is split into three independent components:

```
Volume Variance = (Actual Hours − Budget Hours) × Budget Rate
Rate Variance   = (Actual Rate − Budget Rate)   × Actual Hours
Mix Variance    = Total Variance − Volume Variance − Rate Variance
```

| Brand | Budget | Actual | Total Var | % Over |
|---|---|---|---|---|
| Takealot | R3,636,424 | R3,962,927 | +R326,503 | +9.0% |
| MrD | R2,524,330 | R2,780,915 | +R256,584 | +10.2% |
| TFS | R1,440,328 | R1,577,738 | +R137,410 | +9.5% |
| Sellers | R717,074 | R792,385 | +R75,311 | +10.5% |

Volume variance accounts for >99% of the total overrun — the budget consistently underestimated contact volumes. Rate and mix variance are negligible, confirming the cost model is internally consistent.

---

### Module 3 — Cost Allocation (`src/budget/cost_allocation.py`)

**Approach:** Two allocation methods compared.

Shared service costs (contact centre management, QA, telephony infrastructure) are distributed across brands using:

1. **Simple volume weight** — brand's share of total monthly contacts
2. **Channel-adjusted weight** — volume weight modified by channel intensity (phone = 1.5×, chat = 1.0×, email = 0.7×)

The channel-adjusted method is more defensible in an executive review because it reflects actual infrastructure consumption, not just headcount.

---

### Module 4 — Capacity Planning (`src/budget/capacity_planning.py`)

**Formula:**

```
Required Hours = Contact Volume × Blended AHT (8.1 min) / 60
Gross Hours    = Required Hours / Occupancy Rate
Headcount      = ⌈Gross Hours / Productive Hours per Agent⌉
```

**Assumptions:**
- Productive hours per agent: 132/month (22 days × 8 hrs × 75% productivity)
- Shrinkage: 25%
- Cost per FTE: R28,000/month fully loaded

**Peak headcount requirements (April 2024):**

| Brand | Base | Growth +20% | Stress +40% |
|---|---|---|---|
| Takealot | 7 | 9 | 10 |
| MrD | 5 | 6 | 7 |
| TFS | 3 | 4 | 5 |
| Sellers | 2 | 2 | 2 |
| **Total** | **17** | **21** | **24** |

---

### Module 5 — Executive Dashboard (`src/reporting/executive_dashboard.py`)

A single four-panel figure for senior leadership:

| Panel | Content |
|---|---|
| Top-left | Historical contact volume + 6-month forecast, all brands |
| Top-right | Budget vs actual variance, horizontal waterfall by brand |
| Bottom-left | Cross-brand KPI table (CSAT, FCR, cost per contact, variance %) |
| Bottom-right | CSAT trend over 24 months with Black Friday shading |

Output: `outputs/executive_dashboard.png`

An auto-generated plain-English narrative is saved to `outputs/reports/executive_summary.txt`.

---

## Executive Summary (Auto-Generated)

```
VOLUME & SCALE
  Total contacts handled (24 months, all brands): 299,017
  Strong seasonality confirmed: November Black Friday spike ~40% above baseline.

FINANCIAL PERFORMANCE
  Total actual cost    : R9,113,965
  Total budgeted cost  : R8,318,156
  Total variance       : +R795,809 (+9.6% over budget)
  Primary driver       : Contact volume exceeded budget assumptions across all brands.

CUSTOMER EXPERIENCE
  Average CSAT score   : 7.62 / 10 (target: 7.5)
  Average FCR rate     : 76.2%
  Highest CSAT brand   : Sellers (8.20)
  Lowest CSAT brand    : TFS (7.00)

RECOMMENDATIONS
  1. Revise volume assumptions — consistent underestimation across all brands.
  2. Prioritise chat channel investment — 57% lower cost per contact than phone.
  3. Pre-recruit ahead of November — stress scenario headcount (24 FTE) is the ceiling.
  4. CSAT intervention for TFS — below the 7.5 target floor.
```

---

## Getting Started

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python main.py
```

**Skip data generation (if CSVs already exist):**
```bash
python main.py --skip-data
```

**Run individual modules:**
```bash
python src/data_generation/generate_data.py
python src/forecasting/time_series.py
python src/budget/cost_allocation.py
python src/budget/variance_analysis.py
python src/reporting/executive_dashboard.py
```

**Launch Streamlit app:**
```bash
streamlit run streamlit/app.py
```

---

## Requirements

```
pandas
numpy
matplotlib
statsmodels
streamlit
pyyaml
```

Full list in `requirements.txt`.

---

## Technical Stack

| Layer | Tools |
|---|---|
| Data generation | Python, NumPy, pandas |
| Time series modelling | statsmodels (seasonal_decompose, ExponentialSmoothing) |
| Budget analytics | pandas, NumPy (management accounting decomposition) |
| Visualisation | matplotlib |
| Reporting | matplotlib, auto-generated narrative |
| App layer | Streamlit |
| Version control | Git, GitHub |

---

## Design Decisions

**Why statsmodels over Prophet?**
Holt-Winters is explainable without a PhD, has no external dependencies beyond statsmodels, and is the industry standard for monthly contact volume forecasting. Prophet adds complexity without meaningful accuracy gains on a 24-month series with a clean seasonal pattern.

**Why in-sample validation?**
Holt-Winters with a 12-month seasonal period requires at least two full cycles (24 months) to initialise. A train/test split on 24 months leaves only 18 months in training — insufficient. In-sample validation on the final 6 months is the methodologically correct response to a short series. With more data, a proper out-of-time split would be used.

**Why additive seasonality?**
The seasonal spike size (absolute contacts) is roughly constant year over year rather than growing proportionally with the trend. Additive decomposition reflects this correctly. Multiplicative would be appropriate if the spike size scaled with volume.

**Why three allocation methods compared?**
Simple volume weighting is easy to explain and audit. Channel-adjusted weighting is fairer because phone contacts consume more infrastructure than chat. Showing both makes the recommendation defensible in an executive review.

---

*Built as a portfolio project demonstrating end-to-end CX analytics capability.*
*All data is synthetic. No real customer, financial, or operational data is used.*
