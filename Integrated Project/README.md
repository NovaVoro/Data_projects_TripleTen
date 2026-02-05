# Gold Recovery Prediction Project

This project analyzes and models the gold recovery process using industrial flotation data. It follows the structure of the TripleTen Integrated Project and includes data preparation, feature engineering, model training, evaluation, and final prediction generation through a multi‑page Streamlit application.

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── gold_recovery_train.csv
│   ├── gold_recovery_test.csv
│   └── gold_recovery_full.csv
│
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── metrics.py
│   └── analysis.py
│
├── pages/
│   ├── 1_Data_Overview.py
│   ├── 2_Recovery_Validation.py
│   ├── 3_Feature_Analysis.py
│   ├── 4_Model_Training.py
│   └── 5_Final_Predictions.py
│
└── README.md
```

---

## 📦 Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit App

```bash
streamlit run 1_Data_Overview.py
```

Streamlit will automatically detect the `pages/` directory and expose the multi‑page interface.

---

## 🧹 Data Preparation

The training dataset contains:

- Input features  
- Intermediate cleaner/rougher outputs  
- Final concentrate/tail outputs  
- Calculation columns  
- Target recovery values  

The test dataset contains **only input features**, so the preprocessing pipeline must:

- Drop all output and calculation columns from training  
- Align train/test columns using strict intersection  
- Build a numeric preprocessing pipeline (median imputation + scaling)

Example (from `preprocessing.py`):

```python
cols_to_drop = [
    c for c in df.columns
    if ".output." in c or ".calculation." in c or c in [TARGET_R, TARGET_F]
]
X_train_full = df.drop(columns=cols_to_drop)
```

---

## 🧪 Recovery Formula Validation

The project includes a validation step comparing the provided recovery values with the computed formula:

```python
recovery = C * (F - T) / (F * (C - T))
```

Implemented in `metrics.py` as:

```python
def compute_recovery(feed, conc, tail, df):
    F = df[feed].astype(float)
    C = df[conc].astype(float)
    T = df[tail].astype(float)
    return np.where(F * (C - T) != 0, C * (F - T) / (F * (C - T)), np.nan)
```

---

## 🤖 Model Training

Two models are trained for each target:

- **RandomForestRegressor**
- **LinearRegression**

Each wrapped in a pipeline:

```python
Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(...))
])
```

Evaluation metric:

- **SMAPE** (symmetric mean absolute percentage error)

Weighted score:

```
0.25 * rougher_smape + 0.75 * final_smape
```

The best model is selected automatically.

---

## 📈 Final Predictions

The final Streamlit page:

- Loads the best models  
- Applies them to the aligned test dataset  
- Outputs predicted rougher and final recovery values  
- Displays summary statistics and a preview table  

---

## 🧠 Key Lessons

- Train/test schema mismatch must be handled explicitly  
- Output and calculation columns must be removed from training  
- Preprocessing must be applied consistently through pipelines  
- SMAPE is sensitive to zero denominators—handle with care  
- Streamlit multi‑page apps benefit from modular utilities  

---

## 📜 License

This project is for educational and portfolio purposes.
