# ⛽ Oil Region Profitability Analysis — Streamlit Dashboard

A full end‑to‑end machine learning and risk evaluation dashboard built in **Streamlit**, based on a real‑world oil exploration decision pipeline.

This project evaluates three geographic regions using:

- Data cleaning & deduplication  
- Outlier detection and capping  
- Linear regression modeling  
- Break‑even analysis  
- Bootstrap profit simulation  
- A/B campaign validation  
- Final recommendation for development  

The dashboard is modular, interactive, and designed for portfolio‑grade presentation.

---

## 📁 Project Structure
flowchart TB
    root["📁 streamlit_app"]

    root --> app["📄 app.py<br/>Main Streamlit entry point"]

    root --> data["📂 datasets/"]
    data --> d0["📊 geo_data_0.csv"]
    data --> d1["📊 geo_data_1.csv"]
    data --> d2["📊 geo_data_2.csv"]

    root --> pages["📂 pages/"]
    pages --> p1["📊 Overview"]
    pages --> p2["🧹 Data Cleaning"]
    pages --> p3["🤖 Model Training"]
    pages --> p4["💰 Bootstrap Profit"]
    pages --> p5["🧪 A/B Campaigns"]
    pages --> p6["🏁 Final Recommendation"]

    root --> utils["🛠 utils/"]
    utils --> u1["data_loader.py<br/>Load datasets"]
    utils --> u2["cleaning.py<br/>Data preprocessing"]
    utils --> u3["modeling.py<br/>Model logic"]
    utils --> u4["bootstrap.py<br/>Profit simulation"]
    utils --> u5["visuals.py<br/>Charts & plots"]

## 🚀 Running the App

### 1. Create a virtual environment (recommended)
python -m venv venv

- Activate it

**macOS / Linux**  
source venv/bin/activate  

**Windows**  
venv\Scripts\activate  

---

- Install dependencies
pip install -r requirements.txt  

---

- Launch the Streamlit app
streamlit run app.py  

---

### 2. (Optional) Use the Windows launcher
Execute the `run_app.bat` file.  

---

The dashboard will open in your browser at:  
http://localhost:8501

## 📊 Features

### 1. Overview
- Project description
- Global parameter controls
- Dataset previews

### 2. Data Cleaning
- Deduplication
- Outlier detection & capping
- Region summaries

### 3. Model Training
- Linear regression per region
- RMSE & predicted reserves
- Break‑even comparison
- Predicted vs actual scatterplots

### 4. Bootstrap Profit Simulation
- 1,000+ simulated exploration campaigns
- Profit distribution
- 95% CI
- Loss risk

### 5. A/B Campaign Evaluation
- Campaign A: model training + bootstrap
- Campaign B: independent profit evaluation
- Stability & generalization assessment

### 6. Final Recommendation
- Combined evaluation
- Risk‑filtered region selection
- Executive‑style summary

---

## 🛠 Technologies Used
- Python 3.10+
- Streamlit
- Pandas
- NumPy
- scikit‑learn
- Matplotlib / Seaborn

---

## 📄 License
This project is provided for educational and portfolio purposes.

---

## ✨ Author
Built by **Travis Daily**  
Founder & Creative Director — **NovaVoro Interactive**
