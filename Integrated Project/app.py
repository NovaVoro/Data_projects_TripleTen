import streamlit as st

st.set_page_config(
    page_title="Gold Recovery Prediction",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ Industrial Gold Recovery Prediction Dashboard")
st.markdown("""
Welcome to the interactive dashboard for analyzing and predicting gold recovery
across industrial purification stages.

Use the sidebar to navigate through:
- Data overview  
- Recovery formula validation  
- Feature availability  
- Model training & selection  
- Final predictions  
""")