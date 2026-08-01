import pickle as pk
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide"
)

st.title("🧬 Breast Cancer Risk Assessment Dashboard")
st.markdown(
    "Enter the tumor cell nuclei measurements below to compute real-time prediction probabilities using the trained ML model."
)
st.write("---")


# 2. Cached Artifact Loader
@st.cache_resource
def load_artifacts():
    try:
        with open("breast_cancer_data.pkl", "rb") as f:
            artifacts = pk.load(f)
        return artifacts["model"], artifacts["scaler"]
    except FileNotFoundError:
        st.error(
            "❌ Model file 'breast_cancer_data.pkl' not found! Make sure it is in the working directory."
        )
        return None, None


model, scaler = load_artifacts()

# 3. Sidebar Configuration & Controls
st.sidebar.header("Navigation & Controls")
st.sidebar.info("Model running natively inside Streamlit using loaded PKL artifacts.")

# Decision Threshold Slider
threshold_pct = st.sidebar.slider(
    "Malignant Decision Threshold (%)",
    min_value=1,
    max_value=99,
    value=50,
    help="Default is 50%. Lowering this threshold increases model sensitivity for clinical triage.",
)

# Quick Preset Selector for Easy Testing
st.sidebar.subheader("🧪 Quick Test Presets")
preset_type = st.sidebar.radio(
    "Load Sample Values:",
    ("Typical Benign Patient", "Typical Malignant Patient"),
)

# Set defaults based on selected preset
if preset_type == "Typical Benign Patient":
    defaults = {
        "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, "area_mean": 566.3,
        "smoothness_mean": 0.09779, "compactness_mean": 0.08129, "concavity_mean": 0.06664, "concave_points_mean": 0.04781,
        "symmetry_mean": 0.1885, "fractal_dimension_mean": 0.05766,
        "radius_se": 0.2699, "texture_se": 0.7886, "perimeter_se": 2.058, "area_se": 23.56,
        "smoothness_se": 0.008462, "compactness_se": 0.01460, "concavity_se": 0.02387, "concave_points_se": 0.01315,
        "symmetry_se": 0.01980, "fractal_dimension_se": 0.002300,
        "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.70, "area_worst": 711.2,
        "smoothness_worst": 0.1440, "compactness_worst": 0.1773, "concavity_worst": 0.2390, "concave_points_worst": 0.1288,
        "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.07259
    }
else:
    defaults = {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.80, "area_mean": 1001.0,
        "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001, "concave_points_mean": 0.1471,
        "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
        "radius_se": 1.0950, "texture_se": 0.9053, "perimeter_se": 8.5890, "area_se": 153.40,
        "smoothness_se": 0.006399, "compactness_se": 0.04904, "concavity_se": 0.05373, "concave_points_se": 0.01587,
        "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.60, "area_worst": 2019.0,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119, "concave_points_worst": 0.2654,
        "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.11890
    }

# 4. Input Tabs
tabs = st.tabs(
    ["📊 Mean Metrics", "📈 Standard Error (SE)", "📉 Worst/Largest Metrics"]
)

# Tab 1: Mean Metrics
with tabs[0]:
    st.subheader("Mean Metrics of Cell Nuclei")
    col1, col2, col3 = st.columns(3)
    radius_mean = col1.number_input("Radius Mean", value=defaults["radius_mean"], format="%.4f")
    texture_mean = col2.number_input("Texture Mean", value=defaults["texture_mean"], format="%.4f")
    perimeter_mean = col3.number_input("Perimeter Mean", value=defaults["perimeter_mean"], format="%.4f")

    col4, col5, col6 = st.columns(3)
    area_mean = col4.number_input("Area Mean", value=defaults["area_mean"], format="%.2f")
    smoothness_mean = col5.number_input("Smoothness Mean", value=defaults["smoothness_mean"], format="%.4f")
    compactness_mean = col6.number_input("Compactness Mean", value=defaults["compactness_mean"], format="%.4f")

    col7, col8, col9 = st.columns(3)
    concavity_mean = col7.number_input("Concavity Mean", value=defaults["concavity_mean"], format="%.4f")
    concave_points_mean = col8.number_input("Concave Points Mean", value=defaults["concave_points_mean"], format="%.4f")
    symmetry_mean = col9.number_input("Symmetry Mean", value=defaults["symmetry_mean"], format="%.4f")

    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=defaults["fractal_dimension_mean"], format="%.4f")

# Tab 2: Standard Error Metrics
with tabs[1]:
    st.subheader("Standard Error (SE) Metrics")
    col1, col2, col3 = st.columns(3)
    radius_se = col1.number_input("Radius SE", value=defaults["radius_se"], format="%.4f")
    texture_se = col2.number_input("Texture SE", value=defaults["texture_se"], format="%.4f")
    perimeter_se = col3.number_input("Perimeter SE", value=defaults["perimeter_se"], format="%.4f")

    col4, col5, col6 = st.columns(3)
    area_se = col4.number_input("Area SE", value=defaults["area_se"], format="%.2f")
    smoothness_se = col5.number_input("Smoothness SE", value=defaults["smoothness_se"], format="%.4f")
    compactness_se = col6.number_input("Compactness SE", value=defaults["compactness_se"], format="%.4f")

    col7, col8, col9 = st.columns(3)
    concavity_se = col7.number_input("Concavity SE", value=defaults["concavity_se"], format="%.4f")
    concave_points_se = col8.number_input("Concave Points SE", value=defaults["concave_points_se"], format="%.4f")
    symmetry_se = col9.number_input("Symmetry SE", value=defaults["symmetry_se"], format="%.4f")

    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=defaults["fractal_dimension_se"], format="%.4f")

# Tab 3: Worst/Largest Metrics
with tabs[2]:
    st.subheader("Worst/Largest Metrics")
    col1, col2, col3 = st.columns(3)
    radius_worst = col1.number_input("Radius Worst", value=defaults["radius_worst"], format="%.4f")
    texture_worst = col2.number_input("Texture Worst", value=defaults["texture_worst"], format="%.4f")
    perimeter_worst = col3.number_input("Perimeter Worst", value=defaults["perimeter_worst"], format="%.4f")

    col4, col5, col6 = st.columns(3)
    area_worst = col4.number_input("Area Worst", value=defaults["area_worst"], format="%.2f")
    smoothness_worst = col5.number_input("Smoothness Worst", value=defaults["smoothness_worst"], format="%.4f")
    compactness_worst = col6.number_input("Compactness Worst", value=defaults["compactness_worst"], format="%.4f")

    col7, col8, col9 = st.columns(3)
    concavity_worst = col7.number_input("Concavity Worst", value=defaults["concavity_worst"], format="%.4f")
    concave_points_worst = col8.number_input("Concave Points Worst", value=defaults["concave_points_worst"], format="%.4f")
    symmetry_worst = col9.number_input("Symmetry Worst", value=defaults["symmetry_worst"], format="%.4f")

    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=defaults["fractal_dimension_worst"], format="%.4f")

st.write("---")

# 5. Inference Execution Block
if st.button("🚀 Analyze Patient Features", type="primary", use_container_width=True):
    if model is None or scaler is None:
        st.error("Model artifacts are missing. Cannot compute predictions.")
    else:
        # EXACT COLUMN NAMES matching original CSV seen during scaler.fit()
        feature_dict = {
            "radius_mean": radius_mean,
            "texture_mean": texture_mean,
            "perimeter_mean": perimeter_mean,
            "area_mean": area_mean,
            "smoothness_mean": smoothness_mean,
            "compactness_mean": compactness_mean,
            "concavity_mean": concavity_mean,
            "concave points_mean": concave_points_mean,  # Space preserved
            "symmetry_mean": symmetry_mean,
            "fractal_dimension_mean": fractal_dimension_mean,
            "radius_se": radius_se,
            "texture_se": texture_se,
            "perimeter_se": perimeter_se,
            "area_se": area_se,
            "smoothness_se": smoothness_se,
            "compactness_se": compactness_se,
            "concavity_se": concavity_se,
            "concave points_se": concave_points_se,  # Space preserved
            "symmetry_se": symmetry_se,
            "fractal_dimension_se": fractal_dimension_se,
            "radius_worst": radius_worst,
            "texture_worst": texture_worst,
            "perimeter_worst": perimeter_worst,
            "area_worst": area_worst,
            "smoothness_worst": smoothness_worst,
            "compactness_worst": compactness_worst,
            "concavity_worst": concavity_worst,
            "concave points_worst": concave_points_worst,  # Space preserved
            "symmetry_worst": symmetry_worst,
            "fractal_dimension_worst": fractal_dimension_worst,
        }

        # Format DataFrame to align with fitted scaler
        input_df = pd.DataFrame([feature_dict])

        # Step A: Scale features safely
        scaled_features = scaler.transform(input_df)

        # Step B: Predict class probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        benign_prob = round(float(probabilities[0]) * 100, 2)
        malignant_prob = round(float(probabilities[1]) * 100, 2)

        # Step C: Output Classification & Probabilities
        st.write("## 📋 Diagnostic Analysis Results")

        if malignant_prob >= threshold_pct:
            st.error(
                f"### 🔴 Diagnostic Classification: Malignant (Above {threshold_pct}% Threshold)"
            )
        else:
            st.success(
                f"### 🟢 Diagnostic Classification: Benign (Below {threshold_pct}% Threshold)"
            )

        col_a, col_b = st.columns(2)
        col_a.metric(label="Probability of Benign", value=f"{benign_prob}%")
        col_b.metric(
            label="Probability of Malignant", value=f"{malignant_prob}%"
        )