# Gold Recovery Prediction Dashboard

A modular, multi-page Streamlit dashboard for analyzing and predicting gold recovery from ore processing.  
The app is designed as a portfolio-ready, production-lean interface for exploring process data, model performance, and scenario simulations.

---

## 1. Project overview

This project tackles a classic industrial ML problem: predicting the final recovery of gold from ore based on process parameters measured at different stages of the pipeline.

**Core goals:**

- **Understand** the ore processing pipeline and key features affecting gold recovery.
- **Explore** the data with interactive visualizations and filters.
- **Train and evaluate** regression models to predict final recovery.
- **Simulate scenarios** and compare model predictions under different process conditions.
- **Present** everything through a clean, modular Streamlit dashboard.

---

## 2. Repository structure

A suggested, portfolio-friendly layout:

```text
gold_recovery_project/
├─ data/
│  ├─ raw/
│  │  └─ gold_recovery_full.csv
│  ├─ processed/
│  │  └─ gold_recovery_clean.parquet
│  └─ README.md
├─ notebooks/
│  ├─ 01_eda_gold_recovery.ipynb
│  ├─ 02_feature_engineering.ipynb
│  └─ 03_modeling_gold_recovery.ipynb
├─ app/
│  ├─ pages/
│  │  ├─ 01_📊_Overview.py
│  │  ├─ 02_🔍_Feature_Insights.py
│  │  ├─ 03_🤖_Model_Performance.py
│  │  └─ 04_🧪_Scenario_Simulator.py
│  ├─ utils/
│  │  ├─ data_loader.py
│  │  ├─ plotting.py
│  │  ├─ modeling.py
│  │  └─ config.py
│  ├─ __init__.py
│  └─ 🏭_Gold_Recovery_Dashboard.py
├─ models/
│  ├─ baseline_linear.pkl
│  ├─ random_forest.pkl
│  └─ gradient_boosting.pkl
├─ environment.yml  or  requirements.txt
├─ .gitignore
└─ README.md
```

You can adapt names and emoji prefixes to match your style, but keep the structure modular and predictable.

---

## 3. Data description

The dataset contains process measurements from multiple stages of the gold recovery pipeline, typically including:

- **Input features:**
  - **Ore composition:** e.g., `rougher.input.feed_au`, `rougher.input.feed_ag`, `rougher.input.feed_pb`
  - **Process parameters:** e.g., `rougher.state.floatbank10_a_air`, `rougher.state.floatbank10_a_level`
  - **Intermediate concentrations:** e.g., `rougher.output.concentrate_au`, `primary_cleaner.output.concentrate_au`
  - **Throughput and size metrics:** e.g., `rougher.input.feed_rate`, `rougher.input.feed_size`

- **Targets:**
  - **Rougher recovery:** `rougher.output.recovery`
  - **Final recovery:** `final.output.recovery` (main prediction target)

- **Time index:**
  - A timestamp or sequential index column, e.g., `date` or `timestamp`.

### 3.1. Typical preprocessing steps

- **Drop** columns with excessive missing values or leakage (e.g., direct target duplicates).
- **Align** train/test splits by timestamp to avoid leakage.
- **Impute** missing values using simple strategies (median/mean) or model-based imputation.
- **Engineer** domain-aware features:
  - Ratios of concentrate to feed.
  - Stage-to-stage recovery deltas.
  - Rolling means/volatility for key process variables.

---

## 4. Modeling approach

The project is framed as a regression problem:

- **Target:** `final.output.recovery`
- **Metric:** Symmetric Mean Absolute Percentage Error (sMAPE) or MAE/RMSE, depending on the brief.

### 4.1. Baseline models

- **DummyRegressor:** Predicts a constant (e.g., mean recovery) as a sanity check.
- **Linear models:** `LinearRegression`, `Ridge`, or `Lasso` for interpretability.

### 4.2. Tree-based models

- **RandomForestRegressor**
- **GradientBoostingRegressor** or **XGBoost** (if allowed)
- Optional: **CatBoostRegressor** for strong tabular performance.

### 4.3. Evaluation

- **Train/validation split** by time (no shuffling) to mimic production.
- **Cross-validation** on time-based folds if dataset size allows.
- **Model comparison** via:
  - sMAPE / MAE / RMSE
  - Residual plots
  - Feature importance (permutation or model-based)

---

## 5. Streamlit dashboard

The dashboard is organized into multiple pages under `app/pages/`.  
Each page focuses on a specific part of the workflow.

### 5.1. Main entry point

`app/🏭_Gold_Recovery_Dashboard.py`:

- **Sets global config:** page title, layout, theme.
- **Loads shared utilities:** data loader, cached models, and plotting helpers.
- **Provides navigation:** Streamlit’s built-in multipage navigation (via `pages/` folder).

Example minimal launcher:

```python
import streamlit as st
from app.utils.data_loader import load_processed_data

st.set_page_config(
    page_title="Gold Recovery Dashboard",
    page_icon="🏭",
    layout="wide"
)

@st.cache_data
def get_data():
    return load_processed_data()

def main():
    st.title("🏭 Gold Recovery Prediction Dashboard")
    st.markdown(
        "Explore ore processing data, model performance, and scenario simulations "
        "for predicting final gold recovery."
    )
    df = get_data()
    st.metric("Rows", len(df))
    st.metric("Columns", len(df.columns))

if __name__ == "__main__":
    main()
```

### 5.2. Page: Overview

`app/pages/01_📊_Overview.py`:

- **High-level stats:** row counts, date range, missingness.
- **Global distributions:** histograms, boxplots for key variables.
- **Time series:** recovery over time, feed grade trends.

Key interactions:

- **Filters** by date range or production line.
- **Toggle** between rougher and final recovery views.

### 5.3. Page: Feature insights

`app/pages/02_🔍_Feature_Insights.py`:

- **Correlation heatmaps** for selected subsets of features.
- **Scatter plots** of features vs. recovery.
- **Partial dependence–style views** (if precomputed or approximated).

User controls:

- **Feature selection** dropdowns.
- **Log/linear scale toggles** for skewed variables.
- **Sampling sliders** to avoid overplotting.

### 5.4. Page: Model performance

`app/pages/03_🤖_Model_Performance.py`:

- **Model comparison table** with metrics (sMAPE, MAE, RMSE).
- **Residual plots** and predicted vs. actual charts.
- **Feature importance** bar charts.

User controls:

- **Model selector** (e.g., baseline, RF, GBM).
- **Fold selector** if cross-validation results are stored.
- **Error focus** (e.g., high-error subset exploration).

### 5.5. Page: Scenario simulator

`app/pages/04_🧪_Scenario_Simulator.py`:

- **Interactive sliders** for key process variables (feed grade, air flow, pulp density, etc.).
- **Model prediction** of final recovery for the configured scenario.
- **Comparison** to baseline or historical averages.

User controls:

- **Model choice** for simulation.
- **Preset scenarios** (e.g., “High-grade ore”, “Low air flow”, “Aggressive cleaning”).
- **Export** scenario configuration as JSON or CSV (optional).

---

## 6. Utilities

All shared logic lives in `app/utils/` to keep pages lean and readable.

### 6.1. Data loader

`app/utils/data_loader.py`:

- **Single source of truth** for loading data.
- **Caching** via `st.cache_data` to avoid repeated I/O.
- **Config-driven paths** (read from `config.py`).

Example:

```python
import pandas as pd
from pathlib import Path
from .config import DATA_DIR

def load_processed_data() -> pd.DataFrame:
    path = Path(DATA_DIR) / "processed" / "gold_recovery_clean.parquet"
    df = pd.read_parquet(path)
    return df
```

### 6.2. Plotting helpers

`app/utils/plotting.py`:

- **Encapsulate** all plotting logic (e.g., Plotly or Matplotlib).
- **Consistent styling** across pages.
- **Reusable** functions for histograms, scatter plots, time series, and feature importance.

Example:

```python
import plotly.express as px

def plot_recovery_distribution(df, column="final.output.recovery"):
    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        title="Distribution of Final Recovery",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig
```

### 6.3. Modeling helpers

`app/utils/modeling.py`:

- **Load trained models** from `models/`.
- **Provide a unified predict interface** for the dashboard.
- **Optionally** include a small training pipeline for experimentation (but keep heavy training out of the UI path).

Example:

```python
import joblib
from pathlib import Path
from .config import MODELS_DIR

def load_model(name: str):
    path = Path(MODELS_DIR) / f"{name}.pkl"
    return joblib.load(path)

def predict_recovery(model, X):
    return model.predict(X)
```

### 6.4. Config

`app/utils/config.py`:

- **Centralize paths** and constants.

Example:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "final.output.recovery"
RANDOM_STATE = 42
```

---

## 7. Installation

You can use either `conda` with `environment.yml` or `pip` with `requirements.txt`.

### 7.1. Using conda

```bash
conda env create -f environment.yml
conda activate gold-recovery
```

### 7.2. Using pip

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Typical dependencies:

```text
streamlit
pandas
numpy
scikit-learn
plotly
pyarrow
joblib
```

---

## 8. Running the dashboard

From the project root:

```bash
streamlit run app/🏭_Gold_Recovery_Dashboard.py
```

Or, if you prefer a simpler filename:

```bash
streamlit run app/main.py
```

Then open the local URL printed in the terminal (usually `http://localhost:8501`).

---

## 9. Reproducing the modeling pipeline

If you want to retrain models from scratch:

1. **Run EDA and feature engineering notebooks** in `notebooks/` to:
   - Inspect distributions and correlations.
   - Decide on feature subsets and transformations.
   - Save a processed dataset to `data/processed/`.

2. **Run the modeling notebook**:
   - Train baseline and advanced models.
   - Evaluate using time-based splits.
   - Save the best models to `models/` as `.pkl` files.

3. **Update config**:
   - Ensure `MODELS_DIR` and `TARGET_COLUMN` in `config.py` match your artifacts.

4. **Restart the dashboard**:
   - The app will now use the updated models and data.

---

## 10. Customization ideas

- **Add authentication** for internal deployments (e.g., Streamlit secrets + simple login).
- **Integrate experiment tracking** (e.g., MLflow) and surface run comparisons in a new page.
- **Add alerts** for out-of-distribution inputs in the simulator.
- **Export reports** (PDF/HTML) summarizing model performance and key insights.
- **Dark/light theme toggle** via Streamlit’s theme configuration.

---

## 11. Best practices and design notes

- **Deterministic data loading:** all pages rely on the same cached loader to avoid subtle discrepancies.
- **No heavy training in the UI path:** training is done offline; the app only loads artifacts.
- **Clear separation of concerns:**
  - Pages: layout and user interaction.
  - Utils: data, plotting, modeling, config.
- **Portfolio-ready polish:**
  - Consistent emoji prefixes for pages.
  - Clean, minimal copy on each page.
  - Thoughtful defaults for filters and sliders.

---

