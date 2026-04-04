# SkyCity Auckland — Profit Optimization Platform

**Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations**

A machine learning-powered platform for SkyCity Auckland Restaurants & Bars that provides data-driven recommendations for channel mix optimization, cost management, and profit forecasting across multiple delivery channels.

---

## 🎯 Problem Statement

SkyCity Auckland Restaurants & Bars operates across multiple channels:
- **In-Store** dining
- **Uber Eats** delivery
- **DoorDash** delivery
- **Self-Delivery** operations

Without predictive intelligence, profit optimization decisions rely on intuition rather than data. This platform leverages machine learning to forecast profitability under different channel mixes and provide prescriptive recommendations.

---

## ✨ Key Features

### 📊 Interactive Dashboard (Streamlit)
- **5 integrated modules** for analysis, prediction, and optimization
- Real-time profit forecasting using 4 regression models
- Scenario simulation with instant profit impact visibility
- Channel-mix optimization recommendations with operating bounds
- CSV export for reporting and further analysis

### 🤖 Predictive Models
- **4 model comparison:** Linear Regression, Random Forest, Gradient Boosting, XGBoost
- **25 engineered features** including channel ratios, cost factors, and interaction terms
- **Cross-validation diagnostics** (RMSE, R², MAE) for model reliability
- **Uncertainty quantification** with 95% prediction intervals
- **Residual analysis** for prediction quality assessment

### 🎮 What-If Simulator
- Adjust channel mix (In-Store, Uber Eats, DoorDash, Self-Delivery percentages)
- Modify operational parameters (commission rates, delivery costs, growth factors)
- Real-time profit prediction with 4-model consensus
- Commission sensitivity analysis across delivery channels
- Channel profit breakdown visualization

### 🔧 Optimization Panel
- **Channel Efficiency Analysis** – Profit contribution by channel
- **Commission Analysis** – Commission impact on profitability
- **Self-Delivery Threshold** – Break-even analysis for self-delivery operations
- **Recommendations** – Data-driven channel mix and operational adjustments

### 📈 Exploratory Analysis
- Revenue distribution by cuisine type and restaurant segment
- Cost structure analysis across channel types
- Channel dynamics and relationships
- Correlation matrix for feature relationships
- Subregion performance comparison

---

## 📋 Dataset

- **File:** `SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv`
- **Records:** 1,696 unique restaurants
- **Cuisine Types:** Burgers, Chicken Dishes, Chinese, Indian, Japanese, Kebabs/Mediterranean, Pizza, Thai
- **Segments:** Cafe, QSR, Ghost Kitchen, Full-service
- **Subregions:** North Shore, South Auckland, West Auckland, CBD

---

## 🛠 Tech Stack

- **Backend Framework:** Python 3 with scikit-learn, XGBoost
- **Frontend/Dashboard:** Streamlit
- **Visualization:** Plotly
- **Optimization:** SciPy (SLSQP constrained optimization)
- **Statistical Modeling:** statsmodels
- **API:** FastAPI (optional health checks on port 8001)

---

## 🏗 Architecture

```
SkyCity-Auckland-restaurant-operations/
├── app.py                           # Main Streamlit application (all pages)
├── config.toml                      # Streamlit configuration (port 3000)
├── requirements.txt                 # Python dependencies
├── SkyCity Auckland ....csv         # Source dataset
├── README.md                        # This file
├── backend/
│   └── server.py                    # Optional FastAPI health check
├── frontend/
│   └── package.json                 # Container/deployment configuration
├── models/                          # Trained model artifacts
└── memory/
    ├── PRD.md                       # Product Requirements Document
    ├── ResearchPaperDraft.md        # Research manuscript (in progress)
    └── ResearchPaperDraft.pdf       # Rendered manuscript draft
```

---
## Streamlit Dashboard
-https://skycity-auckland-restaurant-operations-b6smfuyhr6yahcqzbu6fde.streamlit.app/

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd SkyCity-Auckland-restaurant-operations
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The dashboard will open at `http://localhost:3000`

### Configuration
- **Port:** Configured in `config.toml` (default: 3000)
- **Theme:** Light theme (customizable in `config.toml`)
- **Dataset:** Automatically loaded from `SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv`

---

## 📖 Usage Guide

### 1. Overview Page
- High-level KPI summary with key metrics
- Revenue and profit breakdown by cuisine and segment
- Channel mix distribution
- Subregion performance comparison

### 2. Exploratory Analysis
Four tabs for deep-dive data exploration:
- **Distribution & Revenue:** Revenue patterns across segments and cuisines
- **Cost Analysis:** COGS and operating expenses by channel
- **Channel Dynamics:** Channel interaction and overlap patterns
- **Correlation Matrix:** Feature relationships for model insights

### 3. Predictive Models
- Train and compare 4 regression models
- View model performance metrics (RMSE, R², MAE)
- Feature importance rankings
- Predicted vs. actual performance
- Residual diagnostics

### 4. What-If Simulator
- **Channel Mix Sliders:** Adjust delivery channel percentages
- **Operational Parameters:** Modify costs, commissions, and growth factors
- **Real-Time Predictions:** See profit impact instantly across all 4 models
- **Sensitivity Analysis:** Explore commission impact on different channels

### 5. Optimization Panel
- View channel-level profit contributions
- Analyze commission break-even points
- Determine self-delivery viability
- Get prescriptive recommendations for channel mix optimization

---

## 📊 Modeling Approach

### Target Variables
- **Total Monthly Net Profit** – Overall profitability
- **Net Profit per Order** – Unit economics
- **Overall Margin** – Margin percentage analysis
- **Channel-level contributions** – Analytical breakdown by channel

### Features (25 total)
| Category | Examples |
|----------|----------|
| **Channel Shares** | InStoreShare, UE_share, DD_share, SD_share |
| **Operational Costs** | CommissionRate, DeliveryCostPerOrder, DeliveryRadiusKM |
| **Growth & Scale** | GrowthFactor, AOV, Orders, RevenuePerOrder |
| **Cost Structure** | COGS_Rate, OPEX_Rate, CostToRevenue |
| **Interactions** | Commission×UE_share, DeliveryCost×SD_share |
| **Channel Ratios** | InStoreRevRatio, UE_RevRatio, DD_RevRatio, SD_RevRatio |
| **Categorical** | CuisineType, Segment, Subregion |

### Model Comparison
```
Linear Regression    → Interpretability & baseline
Random Forest        → Non-linear patterns & robustness
Gradient Boosting    → Superior generalization with uncertainty quantification
XGBoost              → Advanced performance with feature interactions
```

---

## 📈 Key Outputs

### CSV Exports
- **Scenario Summary:** Full prediction and channel breakdown
- **Sensitivity Analysis:** Commission impact across range of values
- **Model Diagnostics:** Performance metrics and cross-validation results

### Visualizations
- Channel mix optimization charts
- Profit sensitivity heatmaps
- Feature importance rankings
- Model prediction intervals
- Residual diagnostics

---

## ✅ Implementation Status

### Fully Implemented ✅
- Interactive 5-module Streamlit dashboard
- 4-model comparison framework with cross-validation
- Channel-mix what-if simulator
- Constrained optimization with SciPy SLSQP
- 95% prediction intervals for uncertainty quantification
- CSV export functionality
- Exploratory analysis suite

### In Progress 🔄
- Research paper manuscript (see `memory/ResearchPaperDraft.md`)
- Executive summary deliverable

### Planned 📋
- Additional channel-level predictive targets
- Advanced visualization enhancements

---

## 📚 Documentation

- **Product Requirements Document:** [memory/PRD.md](memory/PRD.md)
- **Research Paper Draft:** [memory/ResearchPaperDraft.md](memory/ResearchPaperDraft.md)
- **Technology Stack Details:** See `requirements.txt` for all dependencies

---

## 🔗 Related Files

- **Main App:** `app.py` – Streamlit dashboard with all modules
- **Data:** `SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv`
- **Config:** `config.toml` – Streamlit settings
- **Backend:** `backend/server.py` – Optional health check API

---
