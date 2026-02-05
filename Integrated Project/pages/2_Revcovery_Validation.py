import streamlit as st
from utils.data_loader import load_datasets
from utils.metrics import compute_recovery
from sklearn.metrics import mean_absolute_error

st.title("⚗️ Rougher Recovery Formula Validation")

train_df, _, _ = load_datasets()

feed = "rougher.input.feed_au"
conc = "rougher.output.concentrate_au"
tail = "rougher.output.tail_au"
target = "rougher.output.recovery"

if all(col in train_df.columns for col in [feed, conc, tail, target]):
    calc = compute_recovery(feed, conc, tail, train_df)
    valid = (~calc.isna()) & (~train_df[target].isna())
    mae = mean_absolute_error(train_df.loc[valid, target], calc.loc[valid])

    st.metric("MAE between computed and provided recovery", f"{mae:.4f}")
    st.line_chart(calc)
else:
    st.error("Required columns not found in dataset.")