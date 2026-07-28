import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Criminal Justice Recidivism Dashboard",
    page_icon="⚖️",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "🏠 Home",
        "📊 Model Performance",
        "⚖️ Fairness Analysis",
        "🔄 Bias Mitigation",
        "🧠 Explainable AI",
        "📁 Dataset",
        "ℹ️ About"
    ]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "🏠 Home":

    st.title("⚖️ Criminal Justice Recidivism Dashboard")

    st.markdown("---")

    st.header("Project Overview")

    st.write("""
This dashboard presents an Ethical Artificial Intelligence approach for predicting criminal recidivism.

The dashboard allows users to:

- View machine learning model performance
- Analyse fairness across demographic groups
- Compare bias before and after mitigation
- Explore Explainable AI (SHAP)
- Review dataset information
""")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models", "5")

    with col2:
        st.metric("Fairness Metrics", "8")

    with col3:
        st.metric("Dataset", "COMPAS")
        # -----------------------------
# Model Performance Page
# -----------------------------
elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.markdown("### Baseline Model Results")

    import pandas as pd

    df = pd.read_csv("output/tables/baseline_model_results.csv")

    st.dataframe(df, use_container_width=True)