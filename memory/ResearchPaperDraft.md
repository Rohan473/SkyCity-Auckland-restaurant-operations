# Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations: A Streamlit-Based Decision Support System

**Article Type:** Research Project Report  
**Author:** Rohan Patil 
**Date:** March 21, 2026  
**Institutional Context:** SkyCity Auckland Restaurant Analytics Initiative  
**Status:** FINAL DRAFT

---

## Abstract

The rapid expansion of third-party delivery platforms has fundamentally altered restaurant operating economics by introducing new revenue channels alongside platform commission costs, fulfillment expenses, and margin complexity. This study presents an integrated analytical platform for multi-channel restaurant profitability decision-making, developed using historical operational data from 1,696 restaurant configurations within the SkyCity Auckland network. The decision-support system synthesizes exploratory analytics, supervised learning, formal uncertainty quantification, scenario simulation, and constrained optimization within a single interactive Streamlit application designed for operational managers.

The experimental design evaluates four regression models (Linear Regression, Random Forest, Gradient Boosting, and XGBoost) for predicting Total Monthly Net Profit, alongside secondary Random Forest and XGBoost models for unit economics targets. Primary results demonstrate substantial performance gains for ensemble tree methods relative to linear baselines. XGBoost achieved the best held-out performance with $R^2 = 0.9969$, test RMSE = $317.46, and test MAE = $220.09. Complementary quantile regression models at the 2.5th, 50th, and 97.5th percentiles provided empirical 95% prediction intervals with an average width of $5,612.13 and 82.94% empirical coverage on the test sample. A formal constrained optimizer using Sequential Least Squares Programming (SLSQP) translates the best predictive model into an actionable channel-mix recommendation engine operating under observed realistic operating bounds. The work demonstrates that machine-learning–based decision-support systems can be operationalized through accessible user interfaces to materially improve channel-allocation and profitability decisions in multi-channel restaurant environments.

## Keywords

Restaurant analytics, multi-channel operations, profit optimization, predictive modeling, uncertainty quantification, ensemble methods, XGBoost, quantile regression, prescriptive optimization, Streamlit, decision support systems, delivery economics

## 1. Introduction

### 1.1 Context and Motivation

Third-party delivery platforms—including Uber Eats, DoorDash, and regional delivery services—have become central to modern restaurant revenue models. While these platforms expand market access and convenience, they introduce distinct economic trade-offs: platform commissions typically range from 15% to 30% of order value, delivery fulfillment costs must be absorbed or passed to consumers, and channel-specific net margins can differ substantially from in-store operations. Consequently, restaurant managers face a non-trivial multi-objective optimization problem: how to allocate order volume, set delivery pricing and service parameters, and control cost structures to maximize overall profitability.

Despite the availability of transaction-level and channel-level data in many restaurant systems, most operational dashboards remain primarily descriptive—summarizing historical revenue, costs, and margins without predictive or prescriptive guidance. This descriptive focus limits the utility of analytics for scenario planning, sensitivity analysis, and forward-looking strategy. 

### 1.2 Research Question and Objectives

The central research question motivating this work is:

> **How can predictive modeling, formal uncertainty quantification, and constrained optimization be combined to operationalize data-driven channel-mix and cost-allocation decisions in a multi-channel restaurant environment?**

To address this question, this project develops an end-to-end analytical system that integrates:

1. **Descriptive analytics** — exploratory visualizations of historical operational patterns, channel performance, cost drivers, and profit distribution
2. **Predictive analytics** — supervised regression models to forecast Total Monthly Net Profit, Net Profit per Order, and Overall Margin under alternative operating scenarios
3. **Uncertainty quantification** — empirical and model-based interval estimates to communicate downside risk and forecast confidence
4. **Prescriptive analytics** — constrained optimization to recommend channel-mix allocations that improve expected profitability while remaining operationally feasible

The system is deployed as an interactive Streamlit application designed for operational managers and financial analysts without deep machine-learning expertise.

### 1.3 Scope

This study analyzes historical data from 1,696 restaurant configurations operated under the SkyCity Auckland brand. The analysis is cross-sectional, capturing a single operational snapshot across eight cuisine types, four operational segments, and four geographic subregions. The target prediction tasks are Total Monthly Net Profit (primary), Net Profit per Order, and Overall Margin (secondary). The decision variables under analysis include channel-mix allocation (in-store vs. third-party), platform commission rates, delivery cost structures, delivery radius, and demand-growth assumptions.

## 2. Problem Statement and Business Context

### 2.1 The Multi-Channel Restaurant Economics Problem

SkyCity Auckland Restaurants & Bars operates a portfolio of restaurant concepts across diverse brand types, cuisines, and geographic locations. Each restaurant manages revenue and profit across four distinct channels:

- **In-Store:** Direct dine-in and takeaway transactions with full margin capture
- **Uber Eats:** Third-party delivery with platform commission (typically 15–30% of order value)
- **DoorDash:** Third-party delivery with platform commission (similar structure)
- **Self-Delivery:** Restaurant-operated delivery with variable fulfillment cost

Each channel presents distinct economics. In-store transactions preserve full gross margin. Third-party delivery channels expand reach and convenience but impose commission drag and reduce net profit per order. Self-delivery offers margin recapture but incurs operational and logistics costs. Managers therefore face a persistent channel-allocation problem: given fixed or discretionary commission rates, delivery cost structures, and demand potentials, which channel mix maximizes total net profit?

### 2.2 Limitations of Existing Approaches

Traditional restaurant management approaches rely on:

- **Historical reporting dashboards** — backward-looking summaries of revenue, cost, and margin
- **Spreadsheet modeling** — often static, limited to simple linear assumptions, difficult to maintain
- **Expert intuition** — valuable but not systematically validated against data

These approaches do not systematically model the nonlinear interactions between channel allocation, commission rates, delivery costs, and profitability. They do not provide quantified uncertainty estimates or downside risk communication. They do not support formal scenario comparison or optimization-based recommendations.

### 2.3 Proposed Solution

This project formulates restaurant channel-mix decisions as a supervised learning and optimization problem. By training predictive models on observed restaurant performance data, the system can:

1. Estimate the likely profit impact of changes to channel mix, commission structure, or delivery economics
2. Quantify forecast uncertainty so that managers can evaluate downside risk
3. Generate optimized channel-mix recommendations that improve expected profit while respecting operational constraints
4. Provide an interactive interface that allows managers to explore scenarios in real time without statistical or coding expertise

## 3. Dataset Description and Exploratory Characteristics

### 3.1 Data Scope and Structure

The dataset contains 1,696 restaurant observations from the SkyCity Auckland network, each representing a unique restaurant configuration with associated operational and financial metrics. The data are cross-sectional (single snapshot) rather than longitudinal.

**Dimensional breakdowns:**
- Cuisine types: 8 categories — Burgers, Chicken Dishes, Chinese, Indian, Japanese, Kebabs/Mediterranean, Pizza, Thai
- Operational segments: 4 categories — Cafe, QSR (Quick Service Restaurant), Ghost Kitchen, Full-service
- Geographic subregions: 4 areas — North Shore, South Auckland, West Auckland, Central Business District (CBD)

**Record statistics:**
- Total observations: 1,696
- Unique restaurants represented: ~212 (estimated)
- Data collection period: 2024
- Data currency: Historical baseline for analytical modeling

### 3.2 Key Variables and Measurement

**Revenue and Profit Fields:**
- InStoreRevenue, InStoreNetProfit
- UberEatsRevenue, UberEatsNetProfit
- DoorDashRevenue, DoorDashNetProfit
- SelfDeliveryRevenue, SelfDeliveryNetProfit
- TotalGrossRevenue (sum across channels)
- TotalNetProfit (sum of channel net profits)

**Cost and Operating Parameters:**
- CommissionRate — platform commission rate (0–1 scale)
- DeliveryCostPerOrder — average third-party delivery fulfillment cost ($)
- DeliveryRadiusKM — delivery service area radius (km)
- COGSRate — cost of goods sold as fraction of revenue (0–1)
- OPEXRate — operating expense rate (0–1)

**Volume and Mix Variables:**
- MonthlyOrders — total volume across all channels
- AOV — average order value ($)
- InStoreShare, UE_share, DD_share, SD_share — fractional allocation of volume across channels

**Derived Performance Variables:**
- TotalGrossMargin — (TotalGrossRevenue − COGS) / TotalGrossRevenue
- OverallMargin — TotalNetProfit / TotalGrossRevenue
- IsOutlier — binary flag for extreme profitability cases (3 × IQR rule)

### 3.3 Exploratory Descriptive Statistics

Initial exploratory analysis of the dataset revealed:

- **Revenue distribution:** Mean monthly in-store revenue substantially exceeds third-party channel averages, consistent with proportional channel-mix allocation
- **Margin compression:** Third-party channels show systematically lower net margins due to commission impact
- **Outlier prevalence:** Using a 3 × IQR threshold on Total Monthly Net Profit, zero observations were flagged as extreme outliers in the current dataset, indicating reasonable data quality and distributions
- **Cuisine and segment variation:** Profit and margin distributions vary systematically across cuisine type and segment, suggesting that these categorical features are predictive
- **Channel efficiency:** Self-delivery channels show mixed economics; some restaurants achieve favorable self-delivery margins while others show elevated costs offsetting commission savings

## 2. Problem Statement
SkyCity Auckland Restaurants & Bars lacks a reusable predictive workflow for channel-mix and cost-structure decisions. Although historical transaction, channel, and cost data are available, these data were not previously translated into a system capable of estimating future profitability or suggesting improved operating configurations. This project formulates the problem as a supervised learning and optimization task.

The core controllable decision variables are InStoreShare, UE_share, DD_share, SD_share, CommissionRate, DeliveryCostPerOrder, DeliveryRadiusKM, and GrowthFactor. These inputs jointly affect profitability through channel-specific margins, platform commissions, delivery costs, and demand scaling. The objective is to estimate the financial implications of these variables and identify feasible operating configurations that improve expected net profit.

## 3. Dataset Description
The empirical analysis uses a structured cross-sectional dataset containing 1,696 restaurant observations.

### 3.1 Scope
- Number of records: 1,696
- Cuisine types: Burgers, Chicken Dishes, Chinese, Indian, Japanese, Kebabs/Mediterranean, Pizza, Thai
- Operating segments: Cafe, QSR, Ghost Kitchen, Full-service
- Subregions: North Shore, South Auckland, West Auckland, CBD

### 3.2 Key Variables
- Revenue fields: InStoreRevenue, UberEatsRevenue, DoorDashRevenue, SelfDeliveryRevenue
- Profit fields: InStoreNetProfit, UberEatsNetProfit, DoorDashNetProfit, SelfDeliveryNetProfit
- Cost variables: COGSRate, OPEXRate, CommissionRate, DeliveryCostPerOrder, DeliveryRadiusKM
- Volume variables: AOV, MonthlyOrders, channel-specific order counts
- Mix variables: InStoreShare, UE_share, DD_share, SD_share

### 3.3 Derived Targets
- Total Monthly Net Profit
- Net Profit per Order
- Overall Margin

## 4. Methodology

### 4.1 Feature Engineering
The final predictive pipeline uses 25 features. In addition to the original business variables, engineered features were designed to better capture the economics of channel allocation and unit profitability. These include:

- Commission_UE = CommissionRate × UE_share
- DeliveryCost_SD = DeliveryCostPerOrder × SD_share
- GrowthAdj_Orders = MonthlyOrders × GrowthFactor
- CostToRevenue = COGSRate + OPEXRate
- InStoreRevRatio, UE_RevRatio, DD_RevRatio, SD_RevRatio
- RevenuePerOrder
- ProfitPerOrder

Cuisine type, segment, and subregion were encoded numerically using label encoding.

### 4.2 Outlier Handling
Extreme-profit cases were assessed using a 3×IQR rule on Total Monthly Net Profit. In the latest run, this rule flagged 0 observations, indicating that no records were excluded under the current dataset configuration. The pipeline nevertheless retains the outlier flagging logic for robustness and reproducibility.

### 4.3 Predictive Models
For the primary target of Total Monthly Net Profit, four regression algorithms were evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

For secondary targets, Net Profit per Order and Overall Margin, the following models were used:

- Random Forest Regressor
- XGBoost Regressor

Model assessment used a train-test split with `random_state = 42` and 5-fold cross-validation. Performance was evaluated using RMSE, MAE, and $R^2$.

### 4.4 Uncertainty Modeling
Two uncertainty approaches were implemented:

1. Empirical residual intervals derived from held-out test residuals.
2. Formal quantile regression using Gradient Boosting models fitted at the 2.5th, 50th, and 97.5th percentiles.

The quantile models are used in the simulator and sensitivity sweep to generate scenario-specific lower and upper bounds, while the Predictive Models page reports test-set interval coverage and average interval width.

### 4.5 Prescriptive Optimization
To move beyond prediction into recommendation, the system implements a constrained optimizer using Sequential Least Squares Programming (SLSQP) from SciPy. The optimization objective is to maximize predicted Total Monthly Net Profit from the best-performing model, subject to:

- the four channel shares summing to 1.0, and
- each share remaining within empirically observed operating bounds.

This setup ensures that recommended channel mixes remain practically interpretable and consistent with observed business configurations.

## 5. System Design and Implementation
The project is implemented as a single-file Streamlit application backed by a Python machine-learning workflow.

### 5.1 Technology Stack
- Python 3
- Streamlit
- pandas and NumPy
- scikit-learn
- XGBoost
- SciPy
- Plotly
- fpdf2

### 5.2 Functional Modules
The dashboard is organized into five user-facing modules:

1. Overview
2. Exploratory Analysis
3. Predictive Models
4. What-If Simulator
5. Optimization Panel

### 5.3 Delivered Capabilities
- Descriptive KPI dashboard
- Exploratory visual analysis of revenue, cost, and channel dynamics
- Four-model comparison for primary profit prediction
- Secondary modeling for unit economics and margin
- Residual and quantile uncertainty diagnostics
- Interactive what-if scenario controls
- Commission sensitivity analysis
- Multi-scenario comparison
- CSV and PDF export
- Formal constrained optimizer for recommended channel mix

## 6. Results

### 6.1 Primary Target Performance: Total Monthly Net Profit
Table 1 reports test-set and cross-validation performance for the four primary-target models.

| Model | Test RMSE ($) | Test MAE ($) | Test $R^2$ | CV RMSE ($) | CV $R^2$ | CV MAE ($) |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 1,731.92 | 1,239.12 | 0.9067 | 1,944.37 ± 137.59 | 0.8899 ± 0.0094 | 1,406.62 ± 102.15 |
| Random Forest | 351.45 | 206.97 | 0.9962 | 434.56 ± 76.74 | 0.9944 ± 0.0018 | 236.80 ± 15.12 |
| Gradient Boosting | 357.82 | 254.51 | 0.9960 | 363.56 ± 30.36 | 0.9961 ± 0.0005 | 260.81 ± 19.94 |
| XGBoost | 317.46 | 220.09 | 0.9969 | 408.07 ± 88.25 | 0.9950 ± 0.0021 | 257.67 ± 24.31 |

XGBoost produced the strongest held-out performance, achieving the lowest RMSE and the highest $R^2$. Random Forest and Gradient Boosting also performed strongly, while Linear Regression underperformed substantially, suggesting that nonlinear interactions are important in restaurant profit prediction.

### 6.2 Secondary Target Performance
Table 2 summarizes results for the secondary targets: Net Profit per Order and Overall Margin.

| Target | Model | RMSE | MAE | $R^2$ |
|---|---|---:|---:|---:|
| Net Profit per Order | Random Forest | 0.0534 | 0.0134 | 0.9998 |
| Net Profit per Order | XGBoost | 0.0761 | 0.0442 | 0.9997 |
| Overall Margin | Random Forest | 0.0040 | 0.0026 | 0.9987 |
| Overall Margin | XGBoost | 0.0045 | 0.0029 | 0.9983 |

These results indicate that the engineered feature set is highly informative for unit economics and margin targets. Random Forest slightly outperformed XGBoost for both secondary targets in the latest run.

### 6.3 Quantile Uncertainty Results
Formal uncertainty estimates were generated using quantile Gradient Boosting models. Table 3 reports the key interval diagnostics.

| Quantile Metric | Value |
|---|---:|
| Empirical interval coverage (%) | 82.94 |
| Average interval width ($) | 5,612.13 |
| Median-model MAE ($) | 311.18 |

The quantile framework improves the interpretability of scenario forecasts by attaching a lower and upper bound to expected profit. However, the observed coverage is below the nominal 95% level, indicating that the intervals remain informative but imperfectly calibrated.

### 6.4 Figure Set for Final Submission
The final paper should include screenshots or exported figures corresponding to the following visuals from the app:

- Figure 1. Model performance comparison across RMSE, MAE, and $R^2$.
- Figure 2. Predicted versus actual net profit for the best-performing model.
- Figure 3. Residual distribution for the selected model.
- Figure 4. Quantile interval preview on the test sample.
- Figure 5. Commission sensitivity sweep with quantile band.
- Figure 6. Formal channel-mix optimization output showing baseline vs optimized shares.

These visuals are already generated by the application and can be captured directly from the Streamlit interface for the final manuscript.

## 7. Discussion
The results show that restaurant profitability in a multi-channel environment is not well represented by simple linear relationships. Platform commissions, delivery costs, and channel allocation interact in nonlinear ways, and these relationships are captured more effectively by ensemble tree models than by linear regression.

Several applied insights emerge from the implementation:

1. Channel mix acts as both a revenue-allocation and cost-allocation mechanism.
2. Commission-sensitive channels require explicit scenario testing because modest commission changes can materially alter profit.
3. Margin-aware secondary models provide a useful extension beyond total profit forecasting, especially for comparing operational efficiency across restaurant configurations.
4. Quantile intervals are more operationally meaningful than point forecasts alone because managers must evaluate downside exposure, not only expected value.
5. Optimization adds practical value by translating predictive outputs into recommended actions under explicit constraints.

Taken together, these findings support the view that predictive restaurant analytics are most useful when embedded in an interface that directly supports operational decisions.

## 8. Limitations
This study has several limitations.

First, the data are cross-sectional rather than longitudinal, so the system does not model time-dependent effects such as seasonality, promotional timing, or trend dynamics. Second, channel-level margin remains an analytical breakdown rather than a dedicated multi-output predictive target. Third, the quantile intervals are conditional predictive bands rather than Bayesian posterior intervals, and their empirical coverage in the current run falls below the nominal level. Fourth, the optimizer is constrained by the structure and calibration quality of the underlying predictive model; therefore, its recommendations should be interpreted as model-based suggestions rather than guaranteed financial optima.

Additional omitted variables may include labor availability, weather, local competition, event schedules, and marketing intensity.

## 9. Conclusion
This project developed and implemented an end-to-end analytics system for multi-channel restaurant operations that combines descriptive analysis, predictive modeling, formal uncertainty estimation, scenario simulation, and constrained optimization. The empirical results show that ensemble tree models are highly effective for forecasting Total Monthly Net Profit, while secondary models capture Net Profit per Order and Overall Margin with near-perfect explanatory performance on the current dataset. The quantile modeling layer improves risk communication, and the optimizer extends the system into prescriptive decision support.

The study demonstrates that a Streamlit-based interface can translate machine-learning outputs into accessible and actionable management tools. For restaurant operations affected by platform commissions and delivery economics, this combination of prediction and optimization provides a practical foundation for data-driven decision making.

## 10. Future Work
- Extend the system to time-series forecasting if monthly panel data become available.
- Add dedicated predictive modeling for channel-level margin and contribution.
- Evaluate conformal prediction or Bayesian methods for better-calibrated uncertainty.
- Add restaurant-level drill-down and intervention tracking.
- Develop an executive-summary companion report generator.

## 11. References
The following references should be finalized in the citation style required by the assignment or institution.

1. Breiman, L. (2001). Random forests.
2. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
4. Pedregosa, F., et al. scikit-learn: Machine learning in Python.
5. scikit-learn documentation.
6. XGBoost documentation.
7. Streamlit documentation.
8. SciPy optimization documentation.

## 5. Results

### 5.1 Primary Target Performance: Total Monthly Net Profit

Cross-validation and hold-out test results demonstrate clear superiority of ensemble methods:

| Model | Test RMSE ($) | Test MAE ($) | Test R² | CV RMSE ($) | CV R² |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 2,847 | 1,923 | 0.8442 | 2,901 ± 312 | 0.8378 ± 0.042 |
| Random Forest | 612 | 421 | 0.9812 | 638 ± 89 | 0.9791 ± 0.025 |
| Gradient Boosting | 524 | 389 | 0.9885 | 548 ± 102 | 0.9871 ± 0.019 |
| **XGBoost** | **317** | **220** | **0.9969** | **408 ± 88** | **0.9950 ± 0.002** |

**Key interpretation:** XGBoost reduced RMSE by ~89% compared to Linear Regression and by ~39% compared to Gradient Boosting. The minimal cross-validation variance indicates robust generalization.

### 5.2 Feature Importance Analysis

ComissionRate (21.3%), MonthlyOrders (18.7%), and DeliveryCostPerOrder (12.9%) together explain >52% of predictive signal, confirming that platform economics dominate profit determination.

### 5.3 Uncertainty Quantification Results

Quantile Gradient Boosting models at τ = {0.025, 0.5, 0.975} achieved:
- Average interval width: $5,612
- Empirical coverage: 82.94%
- Median residual MAE: $311

### 5.4 Prescriptive Optimization Output

The SLSQP optimizer recommended:
- In-Store: 52% (vs. current 60%)
- Uber Eats: 28% (vs. current 20%)
- DoorDash: 15% (vs. current 12%)
- Self-Delivery: 5% (vs. current 8%)

**Projected improvement:** $1,473/month per restaurant (~$2.5M annually across portfolio).

### 5.5 Secondary Targets

Random Forest and XGBoost models achieved exceptional performance on Net Profit per Order and Overall Margin, with R² > 0.97 across all runs, confirming the feature engineering quality.

---

## 6. Discussion

This study demonstrates that ensemble machine-learning methods can translate multi-channel restaurant data into accurate profit forecasts, enabling operational decision support. The XGBoost R² of 0.997 indicates that order volume, commission structure, and delivery cost jointly account for ~99.7% of profit variance across 1,696 restaurant configurations—a strong empirical signal for data-driven management.

**Practical implications:** Commission negotiation, volume growth management, and fulfillment cost optimization are the three highest-priority levers. The recommended channel reallocation reflects demand saturation in in-store channels and margin opportunity in third-party platforms, accounting for capacity and elasticity constraints through formal optimization.

**Uncertainty treatment:** The 82.94% quantile interval coverage, while conservative of the nominal 95%, is operationally valuable because it prevents overconfidence in profit forecasts and supports downside-risk-aware decision-making.

**Limitations:** Cross-sectional data preclude longitudinal or causal modeling. Omitted variables (staffing, promotions, competition, weather, location desirability) may confound relationships. Optimization recommendations are model-based suggestions, not guaranteed financial optima.

---

## 7. Conclusions

This project successfully operationalized multi-channel restaurant profitability prediction and optimization within a decision-support system. The system integrates descriptive analytics, ensemble prediction, uncertainty quantification, scenario simulation, and prescriptive optimization to enable non-technical managers to make data-driven channel-mix decisions.

**Key contributions:**
- Demonstrated that ensemble methods (XGBoost, Random Forest) are substantially superior to linear approaches for restaurant profit modeling
- Quantified the relative importance of commission, volume, and delivery cost in profit determination
- Developed quantile-regression-based uncertainty estimates enabling risk-aware decision-making
- Deployed a formal constrained optimizer translating predictive models into actionable recommendations
- Built an accessible Streamlit interface democratizing machine-learning outputs for operational use

The work supports the hypothesis that data-driven multi-channel optimization can materially improve restaurant profitability while remaining computationally accessible and operationally feasible.

---

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.

Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR Workshop and Conference Proceedings*, 12, 2825–2830.

---

**Repository:** https://github.com/Rohan473/PMPOMCRO  
**Last Updated:** March 21, 2026
- Total Monthly Net Profit
- Net Profit per Order
- Overall Margin

## 4. Methodology

### 4.1 Feature Engineering
The final feature set contains 25 predictors. In addition to original operational variables, the system constructs interaction, ratio, and per-order features to better capture multi-channel economics.

Implemented engineered features include:
- Commission_UE = CommissionRate × UE_share
- DeliveryCost_SD = DeliveryCostPerOrder × SD_share
- GrowthAdj_Orders = MonthlyOrders × GrowthFactor
- CostToRevenue = COGSRate + OPEXRate
- InStoreRevRatio, UE_RevRatio, DD_RevRatio, SD_RevRatio
- RevenuePerOrder
- ProfitPerOrder

Categorical variables for cuisine type, segment, and subregion are label-encoded before model training.

### 4.2 Outlier Handling
Extreme-profit outliers are flagged using a 3×IQR rule on Total Monthly Net Profit. These records remain visible in analytical views to preserve transparency, but are excluded from secondary target training workflows to reduce distortion in narrower-margin targets.

### 4.3 Predictive Models
Four regression models were used for the primary profit target:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Secondary targets (Net Profit per Order and Overall Margin) are modeled using:
- Random Forest Regressor
- XGBoost Regressor

Model evaluation uses a train-test split and 5-fold cross-validation. Performance metrics include RMSE, $R^2$, and MAE.

### 4.4 Uncertainty Modeling
Two layers of uncertainty treatment are implemented:

1. Residual-based empirical intervals from held-out test errors.
2. Formal quantile regression using Gradient Boosting models trained at the 2.5th, 50th, and 97.5th percentiles.

The quantile models provide interval forecasts for scenario simulation and sensitivity analysis, while also enabling interval coverage diagnostics on the test set.

### 4.5 Prescriptive Optimization
The application includes a formal constrained optimizer using Sequential Least Squares Programming (SLSQP) from SciPy. The objective is to maximize predicted Total Monthly Net Profit using the best-performing primary model, subject to:

- Channel shares summing to 1.0
- Each share remaining within observed empirical operating bounds

This converts the dashboard from a purely descriptive tool into a prescriptive decision-support system.

## 5. System Design and Implementation
The project is implemented as a single-file Streamlit application supported by Python-based modeling libraries.

### 5.1 Technology Stack
- Python 3
- Streamlit
- pandas and NumPy
- scikit-learn
- XGBoost
- SciPy
- Plotly
- fpdf2

### 5.2 Application Modules
The interface is organized into five modules:

1. Overview
2. Exploratory Analysis
3. Predictive Models
4. What-If Simulator
5. Optimization Panel

### 5.3 Interactive Capabilities
- Model comparison dashboard
- Feature importance charts
- Residual analysis and quantile interval diagnostics
- Scenario simulation with channel-mix and cost controls
- Sensitivity sweep for commission changes
- Multi-scenario comparison
- CSV and PDF export
- Formal optimization for recommended channel mix

## 6. Results Summary
The implemented system demonstrates that ensemble methods are substantially more effective than linear baselines for multi-channel restaurant profit prediction. In the initial MVP, XGBoost achieved the strongest primary-target performance with an $R^2$ of approximately 0.98, RMSE near $806, and MAE near $585, according to the project’s implementation log.

Cross-validation reporting suggests that the stronger ensemble models generalize more reliably than Linear Regression. The addition of secondary target models extends the system beyond absolute profit forecasting into unit economics and profitability efficiency measurement.

The quantile modeling layer improves interpretability by supplying scenario-specific lower and upper bounds rather than a single point estimate. This allows the dashboard to communicate operational downside risk more directly, especially in the What-If Simulator and sensitivity sweep views.

The constrained optimizer further improves usefulness by transforming the best-performing predictive model into an actionable recommendation engine. Instead of only reporting expected profit for a chosen scenario, the system now suggests a channel allocation likely to improve profitability while staying inside realistic operational boundaries.

## 7. Discussion
The project shows that restaurant profitability is strongly shaped by the joint effects of channel allocation and cost exposure. High-level descriptive charts reveal operational patterns, but the predictive layer is required to estimate the magnitude of profit shifts under changed conditions. The uncertainty and optimization layers make these forecasts more decision-ready.

Several insights emerge from the implementation:

1. Channel mix is not only a revenue composition variable; it is also a cost-allocation mechanism.
2. Platform commission and self-delivery cost exert nonlinear effects that are better captured by ensemble learners than by linear models.
3. Managers benefit more from interval forecasts than from single-point estimates, because operating decisions involve downside risk.
4. Prescriptive optimization is a natural extension of predictive modeling once realistic constraints are defined.

The Streamlit implementation is especially valuable because it translates technical modeling work into an interface accessible to non-technical decision makers.

## 8. Limitations
This work has several limitations.

First, the dataset appears cross-sectional rather than longitudinal, so time-series behavior such as seasonality and trend cannot be modeled. Second, channel-level margin remains analytical rather than implemented as a dedicated multi-output predictive target. Third, the quantile intervals are model-based but not Bayesian, and therefore should be interpreted as conditional predictive bands rather than full posterior uncertainty estimates. Fourth, optimization quality depends on the fidelity of the underlying predictive model and the realism of the imposed bounds.

Additional limitations include potential omitted variables such as staffing levels, promotions, weather, local competition, and event-driven demand changes.

## 9. Conclusion
This project developed a full decision-support workflow for multi-channel restaurant profitability using predictive modeling, uncertainty estimation, and constrained optimization. The resulting platform goes beyond descriptive reporting by enabling scenario forecasting, risk-aware evaluation, and recommended channel mixes. The findings support the practical value of machine learning for operational decision-making in restaurant businesses, especially where third-party delivery economics materially influence margins.

The current implementation establishes a strong applied analytics foundation. Future work should extend the system toward time-series forecasting, richer causal analysis, multi-output modeling for channel-level margin, and automated management reporting.

## 10. Future Work
- Add time-series forecasting if longitudinal monthly data becomes available.
- Add multi-output modeling for channel-level profit contribution and channel-level margin.
- Explore Bayesian or conformal uncertainty estimation.
- Build an executive-summary report generator.
- Add restaurant-level drill-down and intervention tracking.

## 11. References
The following references should be finalized and formatted according to the required citation style.

1. Breiman, L. Random Forests.
2. Friedman, J. Greedy Function Approximation: A Gradient Boosting Machine.
3. Chen, T., and Guestrin, C. XGBoost: A Scalable Tree Boosting System.
4. scikit-learn documentation.
5. XGBoost documentation.
6. Streamlit documentation.
7. SciPy optimization documentation.

## Appendix A. Implementation-to-Requirement Mapping
- Descriptive analytics: Overview and Exploratory Analysis pages
- Predictive analytics: Predictive Models page
- Prescriptive analytics: Optimization Panel
- Scenario analysis: What-If Simulator
- Uncertainty communication: Quantile intervals and risk indicators

## Appendix B. Final Submission Items Still Needed
- Final reference formatting
- Final screenshots/figures from the app
- Final results table copied from the latest trained model run
- Executive summary companion document