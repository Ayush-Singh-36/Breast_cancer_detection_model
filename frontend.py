import streamlit as st
import requests
#configure the page layout
st.set_page_config(
    page_title = "Breast Cancer Predictor",
    page_icon = "🩺",
    layout = "wide"
)

st.title("🧬 Breast Cancer Risk Assessment Dashboard")
st.markdown("Enter the tumor cell nuclei measurements below to compute a real-time prediction using our deployed Logistic Regression model.")
st.write("---")

#Organise the 30 input features into clean column layouts
st.sidebar.header("Navigate & Controls")
st.sidebar.markdown("Ensure the backend FastAPI service is running before executing predictions.")
#Group into 3 semantic categories based on the dataset structure
tabs = st.tabs(["📊 Mean Metrics", "📈 Standard Error (SE)", "📉Worst/Largest Metrics"])

#Setting up default/placeholder values to make testing easier
with tabs[0]:
    st.subheader("Mean Metrics of cell nuclei")
    col1, col2, col3 = st.columns(3)
    radius_mean = col1.number_input("Radius Mean", value = 17.99, format = "%.4f")
    texture_mean = col2.number_input("Texture Mean", value = 10.38, format = "%.4f")
    perimeter_mean = col3.number_input("Perimeter Mean", value = 122.80, format = "%.4f")

    col4, col5, col6 = st.columns(3)
    area_mean = col4.number_input("Area Mean", value = 1001.0, format = "%.2f")
    smoothness_mean = col5.number_input("smoothness Mean", value = 0.1184, format = "%.4f")
    compactness_mean = col6.number_input("Compactness Mean", value = 0.2776, format = "%.4f")

    col7, col8, col9 = st.columns(3)
    concavity_mean = col7.number_input("Concavity Mean", value = 0.3001, format = "%.4f")
    concave_points_mean = col8.number_input("Concave Points Mean", value = 0.1471, format = "%.4f")
    symmetry_mean = col9.number_input("Symmetry Mean", value = 0.2419, format = "%.4f")

    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value = 0.0787, format = "%.4f")

with tabs[1]:
    st.subheader("Standard Error (SE) Metrics")
    col1, col2, col3 = st.columns(3)
    radius_se = col1.number_input("Radius SE", value = 1.0950, format = "%.4f")
    texture_se = col2.number_input("Texture SE", value = 0.9053, format = "%.4f")
    perimeter_se = col3.number_input("Perimeter SE", value = 8.5890, format = "%.4f")

    col4, col5, col6 = st.columns(3)
    area_se = col4.number_input("Area SE", value = 153.40, format = "%.2f")
    smoothness_se = col5.number_input("Smoothness SE", value = 0.0063, format = "%.4f")
    compactness_se = col6.number_input("Compactness SE", value = 0.0490, format = "%.4f")

    col7, col8, col9 = st.columns(3)
    concavity_se = col7.number_input("Concavity SE", value = 0.0537, format = "%.4f")
    concave_points_se = col8.number_input("Concave Points SE", value = 0.0158, format = "%.4f")
    symmetry_se = col9.number_input("Symmetry SE", value = 0.0300, format = "%.4f")

    fractal_dimension_se = st.number_input("Fractal Dimension SE", value = 0.0061, format = "%.4f")

with tabs[2]:
    st.subheader("Worst/Largest Metrics")
    col1, col2, col3 = st.columns(3)
    radius_worst = col1.number_input("Radius Worst", value = 25.38, format = "%.4f")
    texture_worst = col2.number_input("Texture Worst", value = 17.33, format = "%.4f") 
    perimeter_worst = col3.number_input("Perimeter Worst", value = 184.60, format = "%.4f")

    col4, col5, col6 = st.columns(3)
    area_worst = col4.number_input("Area Worst", value = 2019.0, format = "%.2f")
    smoothness_worst = col5.number_input("Smoothness Worst", value = 0.1622, format = "%.4f")
    compactness_worst = col6.number_input("Compactness Worst", value = 0.6656, format = "%.4f")

    col7, col8, col9 = st.columns(3)
    concavity_worst = col7.number_input("Concavity Worst", value = 0.7119, format = "%.4f")
    concave_points_worst = col8.number_input("Concave Points Worst", value = 0.2654, format = "%.4f")
    symmetry_worst = col9.number_input("Symmetry Worst", value = 0.1189, format = "%.4f")

    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value = 0.1189, format = "%.4f")

st.write("---")
# 3. Create execution logic when user submits the button
if st.button("🚀 Analyze Patient Features", type="primary", use_container_width=True):
    # Construct the payload structure exactly matching app.py's PatientData model
    payload = {
        "radius_mean": radius_mean, "texture_mean": texture_mean, "perimeter_mean": perimeter_mean, "area_mean": area_mean, "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean, "concavity_mean": concavity_mean, "concave_points_mean": concave_points_mean, "symmetry_mean": symmetry_mean, "fractal_dimension_mean": fractal_dimension_mean,
        "radius_se": radius_se, "texture_se": texture_se, "perimeter_se": perimeter_se, "area_se": area_se, "smoothness_se": smoothness_se,
        "compactness_se": compactness_se, "concavity_se": concavity_se, "concave_points_se": concave_points_se, "symmetry_se": symmetry_se, "fractal_dimension_se": fractal_dimension_se,
        "radius_worst": radius_worst, "texture_worst": texture_worst, "perimeter_worst": perimeter_worst, "area_worst": area_worst, "smoothness_worst": smoothness_worst,
        "compactness_worst": compactness_worst, "concavity_worst": concavity_worst, "concave_points_worst": concave_points_worst, "symmetry_worst": symmetry_worst, "fractal_dimension_worst": fractal_dimension_worst
    }
    
    try:
        # Send post request to FastAPI
        with st.spinner("Processing data through ML model pipeline..."):
           # Note: We'll talk about 'localhost' vs Docker networking soon!
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)            
        if response.status_code == 200:
            result = response.json()
            
            # Show a colored visual card based on diagnosis results
            if result["prediction"] == "Malignant":
                st.error(f"### 🔴 Diagnosis Result: {result['prediction']}")
            else:
                st.success(f"### 🟢  Diagnosis Result: {result['prediction']}")
                
            st.metric(label="Model Prediction Confidence", value=f"{result['confidence']}%")
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error occurred')}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Failed to connect to FastAPI Backend. Is the Uvicorn server running on port 8000?")