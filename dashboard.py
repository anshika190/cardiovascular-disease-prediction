import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Title and Description ---
st.title("Advanced General Health Checkup Dashboard")
st.markdown(
    """
    _Explore and visualize health checkup data with dynamic analytics, visual correlations, and export options._
    """
)

# --- Sidebar Filtering ---
st.sidebar.header("🔍 Filter Options")

# --- Data Loading ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "data", "health_dataset.csv"),
        os.path.join(base_dir, "data.csv"),
        os.path.join(base_dir, "health_dataset.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Could not locate health_dataset.csv in data/ or root directory.")

df = load_data()
st.sidebar.markdown("**Filter dataset as needed:**")

# -- Sidebar Filters --
age_categories = df["Age_Category"].dropna().unique()
selected_age = st.sidebar.multiselect("Age Category", sorted(age_categories), default=age_categories)

sex_options = df["Sex"].dropna().unique()
selected_sex = st.sidebar.multiselect("Sex", sorted(sex_options), default=sex_options)

health_options = df["General_Health"].dropna().unique()
selected_health = st.sidebar.multiselect("General Health", sorted(health_options), default=health_options)

# Condition filters
st.sidebar.markdown("**Health Conditions:**")
cond_cols = ["Heart_Disease","Skin_Cancer","Other_Cancer","Depression","Diabetes","Arthritis"]
selected_cond = {}
for cond in cond_cols:
    opts = df[cond].dropna().unique()
    selected_cond[cond] = st.sidebar.multiselect(cond.replace("_"," "), sorted(opts), default=opts)

# Filter dataset based on sidebar selections
filt = (
    df["Age_Category"].isin(selected_age) &
    df["Sex"].isin(selected_sex) &
    df["General_Health"].isin(selected_health)
)
for cond in cond_cols:
    filt = filt & (df[cond].isin(selected_cond[cond]))
filtered_df = df[filt]

# --- Create Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Visualizations", "Correlations", "Export Data"])

with tab1:
    st.subheader(f"Summary Statistics – {filtered_df.shape[0]} records selected")
    st.dataframe(filtered_df.head(25))
    st.write("Descriptive Statistics (numeric columns):")
    st.write(filtered_df.describe())
    
    # Health Alerts
    st.markdown("### 🚦 Health Risk Alerts")
    alert_rows = filtered_df[(filtered_df['BMI'] > 30) | (filtered_df['Diabetes'] == 'Yes')]
    if not alert_rows.empty:
        st.warning(f"⚠️ {alert_rows.shape[0]} individuals with high BMI (>30) or Diabetes risk detected.")
    else:
        st.success("No current high-risk individuals detected.")

with tab2:
    st.markdown("### BMI Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df['BMI'], bins=15, color='royalblue', kde=True, ax=ax1)
    ax1.set_xlabel("BMI")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)
    
    st.markdown("### General Health Breakdown")
    st.bar_chart(filtered_df["General_Health"].value_counts())
    
    st.markdown("### Diabetes Prevalence")
    st.bar_chart(filtered_df["Diabetes"].value_counts())
    
    st.markdown("### 🍍 Fruit & Vegetable Intake vs General Health")
    tmp = (
        filtered_df.groupby("General_Health")[["Fruit_Consumption","Green_Vegetables_Consumption"]]
        .mean()
        .sort_index()
    )
    st.bar_chart(tmp)

with tab3:
    st.markdown("### 🔬 Correlation Heatmap")
    num_cols = ['Height_(cm)','Weight_(kg)','BMI','Fruit_Consumption','Green_Vegetables_Consumption','FriedPotato_Consumption']
    corr = filtered_df[num_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', ax=ax2)
    st.pyplot(fig2)

    st.markdown("### Mean BMI Grouped by Age and Sex")
    agg_df = (filtered_df.groupby(['Age_Category','Sex'])['BMI']
              .mean()
              .unstack()
              .fillna(0)
              .round(2))
    st.dataframe(agg_df)

with tab4:
    st.markdown("### 📁 Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download as CSV",
        data=csv,
        file_name='filtered_health_data.csv',
        mime='text/csv',
    )

# ---- Footer ----
st.markdown("---")
st.info("**Tip:** Use the filters in the sidebar to slice your data and personalize analyses. Switch between tabs to explore different views.")
st.caption("Advanced Streamlit Health Checkup Dashboard | For demonstration purposes.")
st.warning("?? **Academic & Educational Disclaimer**: This dashboard is created for academic research and college project demonstration. The insights and analyses presented are based on statistical population data and do not substitute for professional medical assessment or diagnostic screening.")
