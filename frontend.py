import pickle as pk
import numpy as np
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
            "❌ Model file 'breast_cancer_data.pkl' not found! Please ensure the trained file is in the same directory."
        )
        return None, None


model, scaler = load_artifacts()

# 3. Navigation Controls
st.sidebar.header("Navigation & Controls")
st.sidebar.info(
    "Model running in standalone mode (FastAPI dependency bypassed)."
)

# 4. Input Feature Form Tabs
tabs = st.tabs(
    ["📊 Mean Metrics", "📈 Standard Error (SE)", "📉 Worst/Largest Metrics"]
)

# Tab 1: Mean Metrics
with tabs[0]:
    st.subheader("Mean Metrics of cell nuclei")
    col1, col2, col3 = st.columns(3)
    radius_mean = col1.number_input(
        "Radius Mean", value=17.99, format="%.4f"
    )
    texture_mean = col2.number_input(
        "Texture Mean", value=10.38, format="%.4f"
    )
    perimeter_mean = col3.number_input(
        "Perimeter Mean", value=122.80, format="%.4f"
    )

    col4, col5, col6 = st.columns(3)
    area_mean = col4.number_input("Area Mean", value=1001.0, format="%.2f")
    smoothness_mean = col5.number_input(
        "Smoothness Mean", value=0.1184, format="%.4f"
    )
    compactness_mean = col6.number_input(
        "Compactness Mean", value=0.2776, format="%.4f"
    )

    col7, col8, col9 = st.columns(3)
    concavity_mean = col7.number_input(
        "Concavity Mean", value=0.3001, format="%.4f"
    )
    concave_points_mean = col8.number_input(
        "Concave Points Mean", value=0.1471, format="%.4f"
    )
    symmetry_mean = col9.number_input(
        "Symmetry Mean", value=0.2419, format="%.4f"
    )

    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean", value=0.0787, format="%.4f"
    )

# Tab 2: Standard Error Metrics
with tabs[1]:
    st.subheader("Standard Error (SE) Metrics")
    col1, col2, col3 = st.columns(3)
    radius_se = col1.number_input("Radius SE", value=1.0950, format="%.4f")
    texture_se = col2.number_input("Texture SE", value=0.9053, format="%.4f")
    perimeter_se = col3.number_input(
        "Perimeter SE", value=8.5890, format="%.4f"
    )

    col4, col5, col6 = st.columns(3)
    area_se = col4.number_input("Area SE", value=153.40, format="%.2f")
    smoothness_se = col5.number_input(
        "Smoothness SE", value=0.0063, format="%.4f"
    )
    compactness_se = col6.number_input(
        "Compactness SE", value=0.0490, format="%.4f"
    )

    col7, col8, col9 = st.columns(3)
    concavity_se = col7.number_input(
        "Concavity SE", value=0.0537, format="%.4f"
    )
    concave_points_se = col8.number_input(
        "Concave Points SE", value=0.0158, format="%.4f"
    )
    symmetry_se = col9.number_input("Symmetry SE", value=0.0300, format="%.4f")

    fractal_dimension_se = st.number_input(
        "Fractal Dimension SE", value=0.0061, format="%.4f"
    )

# Tab 3: Worst/Largest Metrics
with tabs[2]:
    st.subheader("Worst/Largest Metrics")
    col1, col2, col3 = st.columns(3)
    radius_worst = col1.number_input(
        "Radius Worst", value=25.38, format="%.4f"
    )
    texture_worst = col2.number_input(
        "Texture Worst", value=17.33, format="%.4f"
    )
    perimeter_worst = col3.number_input(
        "Perimeter Worst", value=184.60, format="%.4f"
    )

    col4, col5, col6 = st.columns(3)
    area_worst = col4.number_input("Area Worst", value=2019.0, format="%.2f")
    smoothness_worst = col5.number_input(
        "Smoothness Worst", value=0.1622, format="%.4f"
    )
    compactness_worst = col6.number_input(
        "Compactness Worst", value=0.6656, format="%.4f"
    )

    col7, col8, col9 = st.columns(3)
    concavity_worst = col7.number_input(
        "Concavity Worst", value=0.7119, format="%.4f"
    )
    concave_points_worst = col8.number_input(
        "Concave Points Worst", value=0.2654, format="%.4f"
    )
    symmetry_worst = col9.number_input(
        "Symmetry Worst", value=0.1189, format="%.4f"
    )

    fractal_dimension_worst = st.number_input(
        "Fractal Dimension Worst", value=0.1189, format="%.4f"
    )

st.write("---")

# 5. Inference Execution Block
if st.button(
    "🚀 Analyze Patient Features", type="primary", use_container_width=True
):
    if model is None or scaler is None:
        st.error("Model artifacts are missing. Cannot compute predictions.")
    else:
        # Step A: Gather all 30 features sequentially matching training schema
        input_features = [
            # 1. Mean Metrics
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,
            # 2. Standard Error Metrics
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,
            # 3. Worst Metrics
            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst,
        ]

        # Step B: Scale inputs using stored StandardScaler
        raw_array = np.array([input_features])
        scaled_features = scaler.transform(raw_array)

        # Step C: Compute Class Prediction & Class Probabilities
        prediction_code = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]

        benign_prob = round(float(probabilities[0]) * 100, 2)
        malignant_prob = round(float(probabilities[1]) * 100, 2)

        # Step D: Render Visual Diagnostic Output
        if prediction_code == 1:
            st.error("### 🔴 Diagnostic Classification: Malignant")
        else:
            st.success("### 🟢 Diagnostic Classification: Benign")

        # Display probability metrics side-by-side
        col_a, col_b = st.columns(2)
        col_a.metric(
            label="Chance of Benign",
            value=f"{benign_prob}%",
        )
        col_b.metric(
            label="Chance of Malignant",
            value=f"{malignant_prob}%",
        )