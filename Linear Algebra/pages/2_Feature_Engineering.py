import streamlit as st

from utils.data_loader import load_data
from utils.plots import plot_income_per_member_box, plot_scaling_preview


@st.cache_data
def get_data():
    return load_data("data/insurance_us.csv")


st.title("📈 Feature Engineering")

df, df_scaled, feature_names = get_data()

st.subheader("Derived feature: income_per_member")
st.write(
    "Income per family member is computed as `income / (family_members + 1)` "
    "to avoid division by zero."
)
st.dataframe(df[["income", "family_members", "income_per_member"]].head())

st.subheader("Income per member by insurance benefits")
st.pyplot(plot_income_per_member_box(df))

st.subheader("Scaling impact preview")
st.write("MaxAbsScaler is applied to `gender`, `age`, `income`, `family_members`.")
st.pyplot(plot_scaling_preview(df, df_scaled, ["age", "income", "family_members"]))

st.subheader("Scaled feature sample")
st.dataframe(df_scaled[feature_names].head())