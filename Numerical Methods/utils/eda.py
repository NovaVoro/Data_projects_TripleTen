import io
import streamlit as st

def eda_summary(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    st.subheader("DataFrame Info")
    st.text(info_str)