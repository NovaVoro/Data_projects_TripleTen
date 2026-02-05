import streamlit as st

st.set_page_config(
    page_title="Oil Region Profitability Analysis",
    layout="wide"
)

st.title("⛽ Oil Region Profitability Analysis Dashboard")

st.markdown("""
Welcome to the **Oil Region Profitability Analysis Dashboard**, a full end‑to‑end 
machine learning and risk evaluation system built in Streamlit.

Use the navigation menu on the left to explore:

### 📄 Pages
- **Overview** — project description & global parameters  
- **Data Cleaning** — deduplication, outlier handling, region summaries  
- **Model Training** — per‑region regression modeling & break‑even analysis  
- **Bootstrap Simulation** — profit distribution, CI, loss risk  
- **A/B Campaigns** — model stability & generalization testing  
- **Final Recommendation** — board‑ready decision summary  

This dashboard mirrors a real exploration decision pipeline and is fully modular, 
scalable, and production‑ready.
""")

st.info("Use the sidebar to navigate through the analysis workflow.")