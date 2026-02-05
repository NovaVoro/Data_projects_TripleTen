# 📘 Insurance Benefits Analysis & Modeling Dashboard  
A multi‑page **Streamlit application** for exploring, modeling, and demonstrating privacy‑preserving transformations on an insurance dataset.  
This project converts a full exploratory notebook into a clean, modular, production‑ready analytics dashboard.

---

## 🚀 Features

### **1. Exploratory Data Analysis (EDA)**
- Dataset preview & descriptive statistics  
- Missing‑value inspection  
- Target distribution (raw & binary)  
- Correlation heatmap  
- Boxplots by insurance benefit level  
- Histograms & KDE plots  
- Gender‑based comparisons  
- Group‑level aggregations  

### **2. Feature Engineering**
- Derived feature: `income_per_member`  
- MaxAbsScaler transformation  
- Before/after scaling visualization  

### **3. kNN Classification**
- Random baseline classifier  
- kNN performance for k = 1..10  
- Scaled vs unscaled comparison  
- Interactive nearest‑neighbor lookup  

### **4. Custom Linear Regression**
- Closed‑form solution implementation  
- Unscaled vs scaled regression  
- RMSE & R² evaluation  
- Weight inspection  

### **5. Data Obfuscation Demo**
- Random invertible matrix generation  
- Feature transformation & recovery  
- LR performance on obfuscated data  
- Demonstrates reversible privacy‑preserving transformations  

---

## 📁 Project Structure

```
insurance_app/
│
├── app.py
├── requirements.txt
├── run_app.bat
│
├── data/
│   └── insurance_us.csv
│
├── pages/
│   ├── 1_📊_Exploratory_Data_Analysis.py
│   ├── 2_📈_Feature_Engineering.py
│   ├── 3_🤖_KNN_Classification.py
│   ├── 4_📐_Linear_Regression.py
│   └── 5_🔒_Obfuscation_Demo.py
│
└── utils/
    ├── data_loader.py
    ├── plots.py
    ├── modeling.py
    └── obfuscation.py
```

---

## ▶️ Running the App

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Launch Streamlit**
```
streamlit run app.py
```

Or on Windows:
```
run_app.bat
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
seaborn
matplotlib
scikit-learn
```

---

## 🧠 Data Overview

The dataset includes:

| Column | Description |
|--------|-------------|
| `gender` | 0 = female, 1 = male |
| `age` | Customer age |
| `income` | Annual salary |
| `family_members` | Number of family members |
| `insurance_benefits` | Count of benefits received |
| `insurance_benefits_received` | Binary target (benefits > 0) |
| `income_per_member` | Engineered feature |

Scaling is applied using **MaxAbsScaler** to preserve sparsity and relative magnitude.

---

## 🔍 Modeling Summary

### **kNN Classification**
- Evaluated for k = 1..10  
- Compared unscaled vs scaled  
- F1 score + confusion matrix  

### **Linear Regression**
- Custom closed‑form implementation  
- Evaluated on both raw and scaled features  
- RMSE and R² reported  

### **Obfuscation**
- Demonstrates reversible transformation:  
  ```
  X' = X @ P  
  X  = X' @ P⁻¹
  ```
- Shows that LR performance remains identical after obfuscation.

---

## 🎯 Purpose

This project demonstrates:
- Clean modularization of a large notebook  
- Professional Streamlit dashboard architecture  
- Reusable utilities for modeling and visualization  
- Privacy‑preserving transformations for ML workflows  
- Portfolio‑ready structure for data science and ML engineering  

---

## 🧠 Author

Developed by **Travis Daily**  
Founder & Creative Director — NovaVoro Interactive  

Data science, analytics, and interactive systems design.

---

## 📄 License

This project is intended for educational and portfolio use.
