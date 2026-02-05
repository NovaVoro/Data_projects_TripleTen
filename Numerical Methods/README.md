# 🚗 Car Price Prediction — Streamlit Dashboard

A fully interactive Streamlit application that walks through an end‑to‑end machine learning workflow for predicting used car prices.  
This project converts a full Jupyter‑based ML pipeline into a clean, modular, production‑ready web app suitable for GitHub, portfolio use, and real‑world demonstrations.

---

## 🌟 Features

### ✔ Built‑in Sample Dataset
The app loads a prepackaged dataset from `data/car_data.csv`, ensuring the dashboard runs instantly with no user uploads required.

### ✔ Complete ML Workflow
The dashboard implements the full modeling pipeline:

- Data loading with error handling  
- Cleaning and sanity checks  
- Feature engineering (log‑transform, vehicle age)  
- Exploratory Data Analysis  
- Visualizations (histograms, heatmaps, boxplots)  
- Preprocessing pipelines for linear and tree‑based models  
- Model training and RMSE evaluation  
- Model comparison across algorithms  

### ✔ Multiple Regression Models
The app supports:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- LightGBM Regressor (if installed)  

### ✔ Modular Codebase
All logic is organized into a `utils/` package for clarity, maintainability, and easy extension.

---

## 🖥️ App Structure

The dashboard is organized into intuitive tabs:

### **1. Home**
Project overview and dataset description.

### **2. Data Preview**
- Shape  
- Columns  
- Head  
- EDA summary  

### **3. Cleaning & Features**
- Cleaned dataset  
- Engineered features  
- Missingness report  

### **4. Visualizations**
- Histograms  
- Correlation heatmap  
- Boxplots  

### **5. Model Training**
- Select a model  
- Train on cleaned + engineered data  
- View RMSE  

### **6. Model Comparison**
Side‑by‑side RMSE table for all supported models.

---

## 🚀 Getting Started

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/car-price-app.git
cd car-price-app
```

### **2. Run the App (Windows)**

Double‑click:

```
run_app.bat
```

This script:

- Creates a virtual environment (if missing)  
- Installs dependencies  
- Launches the Streamlit app  

### **3. Run Manually (Any OS)**

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
```

---

## 📁 Project Structure

flowchart TB
    root["📁 car_price_app"]

    root --> app["📄 app.py<br/>Main Streamlit application"]
    root --> req["📦 requirements.txt<br/>Python dependencies"]
    root --> run["▶️ run_app.bat<br/>Windows launcher"]
    root --> readme["📘 README.md<br/>Project documentation"]

    root --> data["📂 data/"]
    data --> csv["📊 sample_car_data.csv<br/>Example vehicle dataset"]

    root --> utils["🛠 utils/"]
    utils --> dl["data_loader.py<br/>Load datasets"]
    utils --> cl["cleaning.py<br/>Data cleaning & validation"]
    utils --> ft["features.py<br/>Feature engineering"]
    utils --> eda["eda.py<br/>Exploratory analysis"]
    utils --> vis["visuals.py<br/>Charts & visualizations"]
    utils --> prep["preprocessors.py<br/>Pipelines & transformers"]
    utils --> mdl["modeling.py<br/>Model training & prediction"]

---

## 📊 Model Performance

The dashboard computes RMSE on a validation split for each model.  
LightGBM typically performs best when available, but all models are included for comparison.

---

## 🧩 Extending the Project

You can easily extend this app by:

- Adding hyperparameter tuning  
- Supporting user‑uploaded datasets  
- Adding SHAP or feature importance visualizations  
- Exporting trained models  

The modular `utils/` package makes enhancements straightforward.

---

## 🧠 Author

Developed by **Travis Daily**  
Founder & Creative Director — NovaVoro Interactive  

Data science, analytics, and interactive systems design.

---

## 📜 License

This project is released under the MIT License.
