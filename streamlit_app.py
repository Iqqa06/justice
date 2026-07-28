import streamlit as st
import pandas as pd
from pathlib import Path


# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Criminal Justice Recidivism Dashboard",
    page_icon="⚖️",
    layout="wide"
)
# ---------------------------------
# Project paths
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


# ---------------------------------
# Helper function
# ---------------------------------
def load_csv(filename):
    file_path = OUTPUT_DIR / filename

    if file_path.exists():
        return pd.read_csv(file_path)

    st.error(f"File not found: {filename}")
    st.code(str(file_path))
    return None


# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("⚖️ Criminal Justice")

page = st.sidebar.radio(
    "Select a page",
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


# =================================
# Home page
# =================================
if page == "🏠 Home":

    st.title("⚖️ Criminal Justice Recidivism Dashboard")

    st.subheader("Ethical AI, Fairness and Explainability")

    st.write(
        """
        This dashboard presents the results of a machine-learning project
        for criminal justice recidivism prediction using the COMPAS dataset.

        The dashboard focuses on:

        - Machine-learning model performance
        - Fairness across demographic groups
        - Bias mitigation
        - Explainable Artificial Intelligence
        - Dataset characteristics
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Project", "Recidivism Prediction")

    with col2:
        st.metric("Dataset", "COMPAS")

    with col3:
        st.metric("Focus", "Ethical AI")


# =================================
# Model Performance page
# =================================
elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "This page compares the predictive performance of the baseline machine-learning models."
    )

    df = load_csv("baseline_model_results.csv")

    if df is not None:

        st.subheader("Baseline Model Results")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        text_columns = df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if len(numeric_columns) > 0:

            metric_columns = st.columns(
                min(4, len(numeric_columns))
            )

            for index, column_name in enumerate(
                numeric_columns[:4]
            ):

                maximum_value = df[column_name].max()

                with metric_columns[index]:
                    st.metric(
                        label=f"Best {column_name}",
                        value=f"{maximum_value:.3f}"
                    )

            st.markdown("---")

            st.subheader("Interactive Model Comparison")

            selected_metric = st.selectbox(
                "Choose a performance metric",
                numeric_columns
            )

            if len(text_columns) > 0:

                model_column = text_columns[0]

                chart_df = df[
                    [model_column, selected_metric]
                ].copy()

                chart_df = chart_df.set_index(
                    model_column
                )

                st.bar_chart(chart_df)

            else:

                st.warning(
                    "A model-name column was not detected."
                )

        else:

            st.warning(
                "No numeric performance metrics were detected."
            )

# =================================
# Fairness Analysis page
# =================================
elif page == "⚖️ Fairness Analysis":

    st.title("⚖️ Fairness Analysis")

    st.write(
        """
        This page examines whether model outcomes differ across demographic groups.
        Use the selector below to review overall fairness, race-based results,
        or sex-based results.
        """
    )

    fairness_option = st.selectbox(
        "Select fairness analysis",
        [
            "Overall Fairness",
            "Fairness by Race",
            "Fairness by Sex"
        ]
    )

    if fairness_option == "Overall Fairness":
        fairness_df = load_csv("baseline_fairness_overall.csv")

    elif fairness_option == "Fairness by Race":
        fairness_df = load_csv("baseline_fairness_by_race.csv")

    else:
        fairness_df = load_csv("baseline_fairness_by_sex.csv")

    if fairness_df is not None:

        st.subheader(fairness_option)

        st.dataframe(
            fairness_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        numeric_columns = fairness_df.select_dtypes(
            include="number"
        ).columns.tolist()

        text_columns = fairness_df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if len(numeric_columns) > 0:

            selected_metric = st.selectbox(
                "Choose a fairness metric",
                numeric_columns
            )

            if len(text_columns) > 0:

                group_column = text_columns[0]

                chart_df = fairness_df[
                    [group_column, selected_metric]
                ].copy()

                chart_df = chart_df.set_index(group_column)

                st.subheader(
                    f"{selected_metric} Comparison"
                )

                st.bar_chart(chart_df)

            else:

                st.subheader(
                    f"{selected_metric} Values"
                )

                st.bar_chart(
                    fairness_df[[selected_metric]]
                )

        else:

            st.warning(
                "No numeric fairness metrics were detected in this file."
            )

        with st.expander("How to interpret fairness results"):

            st.write(
                """
                Fairness metrics compare model behavior across demographic groups.

                Large differences between groups may indicate that the model
                does not treat all groups equally.

                These results should be considered together with predictive
                performance, dataset quality, legal requirements, and the
                social context of criminal justice decisions.
                """
            )
# =================================
# Bias Mitigation page
# =================================
elif page == "🔄 Bias Mitigation":

    st.title("🔄 Bias Mitigation")

    st.write("""
    This page compares model performance and fairness
    before and after applying bias mitigation techniques.
    """)

    tab1, tab2 = st.tabs(
        [
            "📊 Performance",
            "⚖️ Fairness"
        ]
    )

    # -----------------------------
    # Performance Comparison
    # -----------------------------
    with tab1:

        st.subheader("Performance Before vs After Mitigation")

        performance_df = load_csv("before_after_performance.csv")

        if performance_df is not None:

            st.dataframe(
                performance_df,
                use_container_width=True,
                hide_index=True
            )

            numeric_columns = performance_df.select_dtypes(
                include="number"
            ).columns.tolist()

            text_columns = performance_df.select_dtypes(
                exclude="number"
            ).columns.tolist()

            if len(numeric_columns) > 0 and len(text_columns) > 0:

                metric = st.selectbox(
                    "Select performance metric",
                    numeric_columns
                )

                chart_df = performance_df[
                    [text_columns[0], metric]
                ].copy()

                chart_df = chart_df.set_index(text_columns[0])

                st.bar_chart(chart_df)

    # -----------------------------
    # Fairness Comparison
    # -----------------------------
    with tab2:

        st.subheader("Fairness Before vs After Mitigation")

        fairness_df = load_csv(
            "before_after_fairness_overall.csv"
        )

        if fairness_df is not None:

            st.dataframe(
                fairness_df,
                use_container_width=True,
                hide_index=True
            )

            numeric_columns = fairness_df.select_dtypes(
                include="number"
            ).columns.tolist()

            text_columns = fairness_df.select_dtypes(
                exclude="number"
            ).columns.tolist()

            if len(numeric_columns) > 0 and len(text_columns) > 0:

                metric = st.selectbox(
                    "Select fairness metric",
                    numeric_columns,
                    key="fairness"
                )

                chart_df = fairness_df[
                    [text_columns[0], metric]
                ].copy()

                chart_df = chart_df.set_index(text_columns[0])

                st.bar_chart(chart_df)

    st.success(
        "Use these comparisons to evaluate whether bias mitigation improved fairness while maintaining acceptable predictive performance."
    )
# =================================
# Explainable AI page
# =================================
elif page == "🧠 Explainable AI":

    st.title("🧠 Explainable Artificial Intelligence")

    st.write(
        """
        This page displays SHAP feature importance results.
        Higher values indicate features that have a stronger influence
        on the model's recidivism predictions.
        """
    )

    shap_df = load_csv("shap_feature_importance.csv")

    if shap_df is not None:

        st.dataframe(
            shap_df,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = shap_df.select_dtypes(
            include="number"
        ).columns.tolist()

        text_columns = shap_df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if len(numeric_columns) > 0 and len(text_columns) > 0:

            feature_column = text_columns[0]
            value_column = numeric_columns[0]

            chart_df = shap_df.set_index(
                feature_column
            )[value_column]

            st.subheader("Feature Importance")

            st.bar_chart(chart_df)


# =================================
# Dataset page
# =================================
elif page == "📁 Dataset":

    st.title("📁 Dataset Information")

    st.write(
        """
        The COMPAS dataset contains information used to study
        criminal recidivism risk prediction.

        This page displays the target-variable distribution.
        """
    )

    target_df = load_csv("target_distribution.csv")

    if target_df is not None:

        st.dataframe(
            target_df,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = target_df.select_dtypes(
            include="number"
        ).columns.tolist()

        text_columns = target_df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if len(numeric_columns) > 0 and len(text_columns) > 0:

            category_column = text_columns[0]
            value_column = numeric_columns[0]

            chart_df = target_df.set_index(
                category_column
            )[value_column]

            st.subheader("Target Distribution")

            st.bar_chart(chart_df)


# =================================
# About page
# =================================
elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.write(
        """
        **Project title:** Criminal Justice Recidivism Prediction

        **Area:** Ethical Artificial Intelligence

        **Dataset:** COMPAS

        **Main objectives:**

        - Compare machine-learning models
        - Evaluate fairness across demographic groups
        - Apply bias-mitigation techniques
        - Explain model predictions using SHAP
        - Present the results through an interactive dashboard
        """
    )

    st.warning(
        """
        This dashboard is intended for academic research and education.
        Machine-learning predictions should not be used as the sole basis
        for real criminal justice decisions.
        """
    )