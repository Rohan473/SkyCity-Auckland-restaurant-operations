# SkyCity Auckland — Profit Optimization Platform
**Created:** 2026-02-18  
**Stack:** Python 3 · Streamlit · scikit-learn · XGBoost · Plotly  
**Serving:** Port 3000 via Supervisor (replaces React frontend)

---

## Problem Statement
SkyCity Auckland Restaurants & Bars lacks predictive intelligence for channel-mix decisions. 
This platform provides ML-driven profit forecasting and prescriptive optimization across 
In-Store, Uber Eats, DoorDash, and Self-Delivery channels.

---

## Dataset
- **File:** `/app/SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv`
- **Records:** 1,696 unique restaurants
- **Cuisine types:** Burgers, Chicken Dishes, Chinese, Indian, Japanese, Kebabs/Mediterranean, Pizza, Thai
- **Segments:** Cafe, QSR, Ghost Kitchen, Full-service
- **Subregions:** North Shore, South Auckland, West Auckland, CBD

---

## Architecture
```
/app/
  app.py                  # Main Streamlit application (all pages)
  .streamlit/config.toml  # Streamlit config (port 3000, light theme)
  requirements.txt        # Python dependencies for app/backend, incl. statsmodels
  SkyCity Auckland ....csv  # Source dataset
  frontend/
    package.json          # Container-oriented start script for Streamlit
  backend/
    server.py             # Minimal FastAPI health check on port 8001
  memory/PRD.md           # This file
  memory/ResearchPaperDraft.md  # In-progress manuscript draft
  memory/ResearchPaperDraft.pdf  # Rendered PDF manuscript draft
```

---

## Scope Status
This PRD reflects the current implementation status in the repository.

- **Implemented now:** Single-file Streamlit dashboard with 5 modules, 4 regression models, cross-validation diagnostics, scenario simulation, uncertainty/risk outputs, CSV export, and prescriptive optimization views.
- **Partially implemented:** Channel-level analytical margin modeling as a dedicated predictive target, research-paper finalization, and executive-summary packaging.
- **Not yet implemented from the full assignment brief:** executive summary deliverable and final submission polishing beyond the dashboard and manuscript draft.

### Assignment Compliance Snapshot
#### Fully met
- Interactive Streamlit dashboard
- Channel-mix what-if controls
- Cost sensitivity analysis
- Optimization recommendation panel
- 4-model comparison using RMSE, R², and MAE
- Cross-validation reporting in the UI
- Residual-based prediction intervals and scenario risk indicators
- CSV export for scenario summary, channel breakdown, and sensitivity sweep

#### Partially met
- **Target variables:** Total Monthly Net Profit, Net Profit per Order, and Overall Margin are now modeled. Channel-level margin / contribution remains analytical rather than modeled as a separate multi-output target.
- **Feature engineering:** Interaction terms and growth/cost features are implemented; revenue-ratio features (`InStoreRevRatio`, `UE_RevRatio`, `DD_RevRatio`, `SD_RevRatio`) and per-order features (`RevenuePerOrder`, `ProfitPerOrder`) are now part of the model training set (25 features total). ✅ Fully met.
- **Prescriptive layer:** formal constrained optimization is now implemented with SciPy SLSQP for channel-mix recommendations under observed operating bounds. Safe operating ranges are surfaced from empirical channel-share quantiles and solver bounds. ✅ Fully met.
- **Uncertainty modeling:** quantile Gradient Boosting models now produce model-based 95% intervals for scenario forecasts, sensitivity sweeps, and interval coverage diagnostics. ✅ Fully met.
- **Environment readiness:** `requirements.txt` exists and `statsmodels` support is now available for Plotly OLS trendlines.

#### Not yet met
- Executive summary deliverable

#### In progress
- Research paper draft in [memory/ResearchPaperDraft.md](memory/ResearchPaperDraft.md)
- Research paper PDF in [memory/ResearchPaperDraft.pdf](memory/ResearchPaperDraft.pdf)

---

## Core Requirements (Static)

### Full Assignment Targets
- **Target Variables:** Total Monthly Net Profit, Net Profit per Order, Channel-Level Margin
- **Decision Variables:** InStoreShare, UE_share, DD_share, SD_share, CommissionRate, DeliveryCostPerOrder, DeliveryRadiusKM, GrowthFactor

### Current MVP Target Coverage
- **Modeled targets:** Total Monthly Net Profit, Net Profit per Order, Overall Margin
- **Analytical visualization only:** Channel-Level Margin / channel profit breakdown

### Pages / Modules
1. **Overview** – KPI cards, cuisine revenue, segment profit, channel mix donut, subregion comparison, key observations
2. **Exploratory Analysis** – 4 tabs: Distribution & Revenue | Cost Analysis | Channel Dynamics | Correlation Matrix
3. **Predictive Models** – Train 4 models (Linear Regression, Random Forest, Gradient Boosting, XGBoost), compare RMSE/R²/MAE, feature importance, predicted vs actual, residuals
4. **What-If Simulator** – Channel mix sliders, cost/ops sliders, restaurant profile, real-time 4-model prediction, channel breakdown, commission sensitivity sweep
5. **Optimization Panel** – 4 tabs: Channel Efficiency | Commission Analysis | Self-Delivery Threshold | Recommendations

### ML Features (25 features)
- Channel shares, commission, delivery cost, radius, growth factor, AOV, orders, COGS rate, OPEX rate
- Interaction terms: Commission×UE_share, DeliveryCost×SD_share, GrowthAdj_Orders, CostToRevenue
- Revenue-ratio features: InStoreRevRatio, UE_RevRatio, DD_RevRatio, SD_RevRatio
- Per-order features: RevenuePerOrder, ProfitPerOrder
- Encoded: CuisineType, Segment, Subregion

---

## Implementation History

### 2026-02-18 — MVP
- Built full Streamlit app (`app.py`) with 5 pages, 19 ML features
- All 4 ML models working: LR R²=0.89, RF R²=0.97, GB R²=0.98, XGB R²=0.98
- XGBoost best model with RMSE=$806, R²=0.9798, MAE=$585
- Streamlit configured to run on port 3000 replacing React frontend
- All charts use Plotly with light/clean enterprise theme

### 2026-03-18 — P1 Feature Completion
- Added model persistence: `save_models_to_disk()` / `load_models_from_disk()` using `joblib`; Predictive Models page auto-loads from `models/trained_models.pkl` on fresh sessions; "Re-train" button forces refresh
- Added PDF export using `fpdf2`: `generate_scenario_pdf()` produces a 3-section PDF (parameters, model forecasts with intervals, channel breakdown); download button in What-If Simulator
- Added multi-scenario comparison: "Save Scenario" button stores up to 5 scenarios in `session_state`; renders comparison table and best-model bar chart; "Clear All" resets
- Added secondary ML targets: `prepare_ml_target()` + `train_secondary_models()` (RF, XGBoost) for Net Profit per Order and Overall Margin; shown as new section in Predictive Models page with Predicted vs Actual charts
- Added outlier flagging: 3×IQR on TotalNetProfit in `load_data()`; `IsOutlier` column added; info banner in Overview; outliers excluded from secondary target training
- Added `fpdf2>=2.7,<3.0` to requirements.txt; `HAS_FPDF` feature flag for graceful degradation
- Expanded feature set from 19 to 25 with revenue-ratio and per-order predictors; What-If Simulator scenario rows updated to match training features
- Added formal uncertainty modeling with quantile Gradient Boosting (lower/median/upper models), coverage diagnostics in Predictive Models, and quantile-based intervals in What-If + sensitivity sweep
- Added formal channel-mix optimization using SciPy SLSQP with equality/bound constraints, best-model objective, baseline-vs-optimized comparison, and observed safe-range display

### 2026-03-18 — Research Paper Draft Started
- Created initial manuscript draft in [memory/ResearchPaperDraft.md](memory/ResearchPaperDraft.md)
- Rendered PDF version in [memory/ResearchPaperDraft.pdf](memory/ResearchPaperDraft.pdf)
- Draft includes abstract, introduction, problem statement, dataset description, methodology, implementation, results summary, discussion, limitations, conclusion, future work, and references placeholders
- Remaining paper work is final result-table insertion, figure/screenshots, citation formatting, and submission-style polishing

### 2026-03-14 — Analytics & Delivery Upgrade
- Added 5-fold cross-validation summary to model comparison view
- Added residual-based 95% prediction intervals to scenario forecasts
- Added scenario risk indicator and confidence band on commission sensitivity sweep
- Added CSV exports for scenario summary, channel breakdown, and sensitivity sweep
- Added `requirements.txt` for reproducible environment setup
- Installed `statsmodels` support and made OLS trendlines degrade gracefully when unavailable
- Hardened session-state refresh so older cached model results do not break new diagnostics

---

## Prioritized Backlog

### P0 (Core — Done)
- [x] Dataset loading & feature engineering
- [x] Overview page with KPIs
- [x] EDA with 4 tabs
- [x] 4 ML models with comparison
- [x] What-If Simulator with sliders
- [x] Optimization Panel with recommendations

### P1 (Next)
- [ ] Draft executive summary artifact
- [ ] Finalize research paper references, figures, and final results tables

### P1 (Completed Recently)
- [x] Export scenario results to CSV
- [x] Cross-validation scores for models
- [x] Confidence intervals on predictions
- [x] Add explicit risk indicators and uncertainty communication in the simulator
- [x] Add reproducible Python dependency file (`requirements.txt`)
- [x] Support Plotly OLS trendlines via `statsmodels` when available
- [x] Persist trained models to disk with `joblib` (auto-load on page visit, Re-train button)
- [x] Export scenario results to PDF via `fpdf2`
- [x] Multi-scenario comparison — save up to 5 scenarios, side-by-side table + bar chart
- [x] Secondary ML targets — RF and XGBoost for Net Profit per Order and Overall Margin
- [x] Outlier flagging — 3×IQR on TotalNetProfit in `load_data()`, banner in Overview, excluded from secondary targets

### P2 (Future)
- [ ] Time-series forecasting (if monthly data becomes available)
- [ ] Individual restaurant drill-down page
- [ ] Monte Carlo simulation for risk analysis
- [ ] Custom commission negotiation tool
- [ ] PDF executive report generator
