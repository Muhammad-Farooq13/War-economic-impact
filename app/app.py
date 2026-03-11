"""
app.py
──────
Streamlit web application for the War Economic Impact Predictor.

Features:
  • Interactive form to input conflict characteristics
  • Real-time GDP impact regression prediction
  • Economic severity classification (Mild / Moderate / Severe / Catastrophic)
  • Pre-loaded dataset overview & key charts
  • SHAP-based model explainability panel

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# isort:off
from src.utils import load_config  # noqa: E402
from src.visualization.visualize import (  # noqa: E402
    plot_gdp_by_conflict_type,
    plot_gdp_by_region,
    plot_gdp_distribution,
    plot_severity_distribution,
    plotly_gdp_choropleth,
    plotly_inflation_boxplot,
)

# isort:on

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="War Economic Impact Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
cfg = load_config(ROOT / "config" / "config.yaml")
DATA_CFG = cfg["data"]
MODEL_DIR = ROOT / cfg["paths"]["model_dir"]

SEVERITY_LABELS = {0: "🟢 Mild", 1: "🟠 Moderate", 2: "🔴 Severe", 3: "⚫ Catastrophic"}
SEVERITY_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336", 3: "#212121"}


# ─── Helpers ──────────────────────────────────────────────────────────────────


@st.cache_resource
def load_models():
    reg_path = MODEL_DIR / "regression_xgb.joblib"
    cls_path = MODEL_DIR / "classification_xgb.joblib"
    scaler_path = MODEL_DIR / "scaler_xgb.joblib"

    if not reg_path.exists():
        return None, None, None
    return (
        joblib.load(reg_path),
        joblib.load(cls_path),
        joblib.load(scaler_path),
    )


@st.cache_data
def load_dataset():
    raw_path = ROOT / cfg["paths"]["raw_data"]
    if raw_path.exists():
        return pd.read_csv(raw_path, low_memory=False)
    return None


@st.cache_data
def load_features():
    feat_path = ROOT / "data" / "processed" / "war_economic_features.parquet"
    if feat_path.exists():
        return pd.read_parquet(feat_path)
    return None


def _build_input_vector(inputs: dict, feat_cols: list[str]) -> np.ndarray:
    """Map UI inputs to model feature vector."""
    vec = pd.Series(0.0, index=feat_cols)
    mapping = {
        "Pre_War_Unemployment_%": inputs["pre_unemp"],
        "During_War_Unemployment_%": inputs["pre_unemp"] + inputs["unemp_spike"],
        "Unemployment_Spike_Percentage_Points": inputs["unemp_spike"],
        "Youth_Unemployment_Change_%": inputs["youth_unemp"],
        "Pre_War_Poverty_Rate_%": inputs["pre_poverty"],
        "During_War_Poverty_Rate_%": inputs["pre_poverty"] + inputs["poverty_increase"],
        "Inflation_Rate_%": inputs["inflation"],
        "Currency_Devaluation_%": inputs["currency_deval"],
        "Food_Insecurity_Rate_%": inputs["food_insecurity"],
        "Extreme_Poverty_Rate_%": inputs["extreme_poverty"],
        "Currency_Black_Market_Rate_Gap_%": inputs["bm_gap"],
        "Conflict_Duration_Years": inputs["duration"],
        "Black_Market_Activity_Level": {"Low": 0, "Moderate": 1, "High": 2, "Dominant": 3}[
            inputs["bm_level"]
        ],
        "War_Profiteering_Documented": 1 if inputs["profiteering"] == "Yes" else 0,
        "Informal_Economy_Size_Pre_War_%": inputs["informal_pre"],
        "Informal_Economy_Size_During_War_%": inputs["informal_during"],
    }
    for k, v in mapping.items():
        if k in vec.index:
            vec[k] = v

    # Engineered features
    if "Informal_Economy_Growth_%" in vec.index:
        vec["Informal_Economy_Growth_%"] = inputs["informal_during"] - inputs["informal_pre"]
    if "Economic_Stress_Index" in vec.index:
        stress = (
            inputs["unemp_spike"]
            + inputs["pre_poverty"]
            + inputs["inflation"] / 100
            + inputs["food_insecurity"]
        )
        vec["Economic_Stress_Index"] = min(stress / 200.0, 1.0)

    return vec.values.reshape(1, -1)


# ─── Sidebar ──────────────────────────────────────────────────────────────────


def sidebar() -> dict:
    st.sidebar.title("🌍 Conflict Parameters")
    st.sidebar.markdown("*Adjust inputs to generate predictions*")
    st.sidebar.divider()

    with st.sidebar.expander("📌 Conflict Metadata", expanded=True):
        conflict_type = st.selectbox(
            "Conflict Type",
            [
                "Civil War",
                "World War",
                "Asymmetric War",
                "Interstate/Counter-insurgency",
                "Proxy War",
                "Territorial Dispute",
            ],
        )
        region = st.selectbox(
            "Region",
            [
                "Middle East",
                "Europe",
                "South Asia",
                "East Asia",
                "Sub-Saharan Africa",
                "Latin America",
                "North Africa",
            ],
        )
        status = st.selectbox("Status", ["Ongoing", "Resolved"])
        duration = st.slider("Conflict Duration (years)", 0, 30, 3)

    with st.sidebar.expander("📊 Labour Market", expanded=True):
        pre_unemp = st.slider("Pre-war Unemployment (%)", 0.0, 30.0, 8.0, 0.1)
        unemp_spike = st.slider("Unemployment Spike (pp)", 0.0, 50.0, 15.0, 0.5)
        youth_unemp = st.slider("Youth Unemployment Change (%)", 0.0, 60.0, 20.0, 0.5)

    with st.sidebar.expander("💰 Economy & Finance"):
        inflation = st.slider("Inflation Rate (%)", 0.0, 200.0, 40.0, 1.0)
        currency_deval = st.slider("Currency Devaluation (%)", 0.0, 500.0, 50.0, 5.0)
        bm_gap = st.slider("Black Market Rate Gap (%)", 0.0, 600.0, 50.0, 5.0)
        bm_level = st.selectbox("Black Market Activity", ["Low", "Moderate", "High", "Dominant"])
        profiteering = st.radio("War Profiteering Documented", ["No", "Yes"], horizontal=True)

    with st.sidebar.expander("🏚 Poverty & Food"):
        pre_poverty = st.slider("Pre-war Poverty Rate (%)", 0.0, 60.0, 15.0, 0.5)
        poverty_increase = st.slider("Poverty Rate Increase (pp)", 0.0, 40.0, 10.0, 0.5)
        extreme_poverty = st.slider("Extreme Poverty Rate (%)", 0.0, 30.0, 8.0, 0.5)
        food_insecurity = st.slider("Food Insecurity Rate (%)", 0.0, 50.0, 15.0, 0.5)

    with st.sidebar.expander("🔄 Informal Economy"):
        informal_pre = st.slider("Informal Economy Pre-war (%)", 0.0, 70.0, 25.0, 0.5)
        informal_during = st.slider("Informal Economy During War (%)", 0.0, 100.0, 45.0, 0.5)

    return dict(
        conflict_type=conflict_type,
        region=region,
        status=status,
        duration=duration,
        pre_unemp=pre_unemp,
        unemp_spike=unemp_spike,
        youth_unemp=youth_unemp,
        inflation=inflation,
        currency_deval=currency_deval,
        bm_gap=bm_gap,
        bm_level=bm_level,
        profiteering=profiteering,
        pre_poverty=pre_poverty,
        poverty_increase=poverty_increase,
        extreme_poverty=extreme_poverty,
        food_insecurity=food_insecurity,
        informal_pre=informal_pre,
        informal_during=informal_during,
    )


# ─── Main App ─────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("🌍 War Economic Impact Predictor")
    st.markdown("""
        > *Predicting the economic damage of armed conflicts using pre-war indicators
        > and machine learning. Built on 100,000+ conflict-economic data points.*
        """)

    inputs = sidebar()
    reg_model, cls_model, scaler = load_models()
    df_raw = load_dataset()
    df_feat = load_features()

    tabs = st.tabs(["🔮 Prediction", "📊 Data Explorer", "📈 Model Insights"])

    # ── Tab 1: Prediction ─────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Economic Impact Prediction")

        if reg_model is None:
            st.warning(
                "⚠️ No trained model found. Run the pipeline first:\n\n"
                "```bash\n"
                "make pipeline\n"
                "```"
            )
        else:
            feat_cols = (
                [
                    c
                    for c in df_feat.columns
                    if c not in {DATA_CFG["target_regression"], DATA_CFG["target_classification"]}
                ]
                if df_feat is not None
                else []
            )

            if st.button("🚀 Predict Economic Impact", type="primary", use_container_width=True):
                x_vec = _build_input_vector(inputs, feat_cols)
                x_sc = scaler.transform(x_vec)

                gdp_pred = float(reg_model.predict(x_sc)[0])
                severity_pred = int(cls_model.predict(x_sc)[0])
                severity_label = SEVERITY_LABELS[severity_pred]
                severity_color = SEVERITY_COLORS[severity_pred]

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Predicted GDP Change",
                    f"{gdp_pred:.1f}%",
                    delta=f"{gdp_pred:.1f}pp vs baseline",
                    delta_color="inverse",
                )
                col2.metric(
                    "Economic Severity",
                    severity_label.replace("🟢 ", "")
                    .replace("🟠 ", "")
                    .replace("🔴 ", "")
                    .replace("⚫ ", ""),
                )
                col3.metric("Inflation Input", f"{inputs['inflation']:.1f}%")

                st.markdown(
                    f"""
                    <div style='
                        background-color:{severity_color}22;
                        border-left: 5px solid {severity_color};
                        padding: 16px;
                        border-radius: 6px;
                        margin-top: 10px;
                    '>
                    <h3 style='color:{severity_color}'>{severity_label}</h3>
                    <p>Estimated GDP change: <strong>{gdp_pred:.2f}%</strong></p>
                    <p>This conflict is predicted to cause a <em>{severity_label.split()[-1].lower()}</em>
                    economic disruption based on the provided indicators.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Gauge chart
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=gdp_pred,
                        delta={"reference": 0},
                        title={"text": "GDP Change (%)"},
                        gauge={
                            "axis": {"range": [-100, 20]},
                            "bar": {"color": severity_color},
                            "steps": [
                                {"range": [-100, -50], "color": "#B71C1C"},
                                {"range": [-50, -25], "color": "#F44336"},
                                {"range": [-25, -10], "color": "#FF9800"},
                                {"range": [-10, 20], "color": "#4CAF50"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Data Explorer ──────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Dataset Explorer")
        if df_raw is None:
            st.error("Raw dataset not found in data/raw/")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(df_raw):,}")
            col2.metric("Features", str(len(df_raw.columns)))
            col3.metric("Unique Conflicts", str(df_raw["Conflict_Name"].nunique()))
            col4.metric("Regions", str(df_raw["Region"].nunique()))

            st.markdown("---")

            # Preview
            with st.expander("📋 Raw Data Sample"):
                st.dataframe(df_raw.head(100), use_container_width=True)

            # Add severity label for viz
            df_viz = df_raw.copy()
            df_viz["Conflict_Duration_Years"] = (df_viz["End_Year"] - df_viz["Start_Year"]).clip(
                lower=0
            )
            conditions = [
                df_viz["GDP_Change_%"] > -10,
                (df_viz["GDP_Change_%"] <= -10) & (df_viz["GDP_Change_%"] > -25),
                (df_viz["GDP_Change_%"] <= -25) & (df_viz["GDP_Change_%"] > -50),
                df_viz["GDP_Change_%"] <= -50,
            ]
            df_viz["Severity_Label"] = np.select(conditions, [0, 1, 2, 3], default=0)

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**GDP Change Distribution**")
                fig = plot_gdp_distribution(df_viz, save=False)
                st.pyplot(fig)

            with chart_col2:
                st.markdown("**Severity Label Distribution**")
                fig = plot_severity_distribution(df_viz, save=False)
                st.pyplot(fig)

            st.markdown("**GDP Change by Conflict Type**")
            fig = plot_gdp_by_conflict_type(df_viz, save=False)
            st.pyplot(fig)

            st.markdown("**GDP Change by Region**")
            fig = plot_gdp_by_region(df_viz, save=False)
            st.pyplot(fig)

            st.markdown("**Mean GDP Change by Region (Interactive)**")
            st.plotly_chart(plotly_gdp_choropleth(df_viz), use_container_width=True)

            st.markdown("**Inflation Rate by Conflict Type**")
            st.plotly_chart(plotly_inflation_boxplot(df_viz), use_container_width=True)

    # ── Tab 3: Model Insights ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Model Insights")

        fig_dir = ROOT / cfg["paths"]["figures_dir"]

        shap_bar = fig_dir / "shap_regression_xgb.png"
        shap_bee = fig_dir / "shap_beeswarm_xgb.png"
        reg_eval = fig_dir / "regression_xgb.png"
        conf_mat = fig_dir / "confusion_xgb.png"
        fi_reg = fig_dir / "shap_classification_xgb.png"

        def show_img(path, caption):
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"Run the evaluation notebook to generate: `{path.name}`")

        col1, col2 = st.columns(2)
        with col1:
            show_img(reg_eval, "Actual vs Predicted GDP Change — XGBoost")
        with col2:
            show_img(conf_mat, "Confusion Matrix — Severity Classifier (XGBoost)")

        col3, col4 = st.columns(2)
        with col3:
            show_img(shap_bar, "SHAP Global Feature Importance (Regression)")
        with col4:
            show_img(shap_bee, "SHAP Beeswarm Plot")

        show_img(fi_reg, "SHAP Feature Importance — XGBoost Classifier")

        st.markdown("""
            ---
            ### How to reproduce this analysis

            ```bash
            # 1. Install dependencies
            pip install -r requirements-dev.txt

            # 2. Run full ML pipeline
            make pipeline

            # 3. Launch this app
            make app
            ```

            **Models trained:** XGBoost · LightGBM · Random Forest · Ridge / Logistic Regression  
            **Tuning:** Optuna (50 trials, XGBoost)  
            **Tracking:** MLflow @ `http://localhost:5001`  
            **Explainability:** SHAP TreeExplainer (global bar + beeswarm)
            """)


if __name__ == "__main__":
    main()
