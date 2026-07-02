#importing all the libraries and modules
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle as pk
import numpy as np
import os
from typing import Any

#initialize the FastAPI application
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="A FastAPI production wrapper for the Logistic Regression classification model.",
    version="1.0.0"
)

#Define the expected data structure for incoming requests
class PatientData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float

    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float

    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

#your response schema class 
class PredictionResponse(BaseModel):
        prediction_code: int
        prediction: str
        confidence: float

# Initialize model globally so endpoints can access it
model = None
scaler = None
artifacts = None
MODEL_PATH = "breast_cancer_data.pkl"

# Model loading logic execution
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        artifacts = pk.load(f)
        model = artifacts['model']
        scaler = artifacts['scaler']
    print("✅ Success: Model and Scaler loaded successfully.") 
else:
    print(f"❌Warning: '{MODEL_PATH}' not foundin this directory")
    print(f"Current Working Directory is: {os.getcwd()}")

@app.post("/predict", response_model=PredictionResponse)
def predict_cancer(data: PatientData) -> dict[str, Any]:
    #ensure the model is loaded
    if model is None or scaler is None: 
        raise HTTPException(status_code=503, detail="Model Unavailable")

    try:
        # Gather exactly 30 features in sequential order
        input_features = [
            # 1. Mean Metrics (10 features)
            data.radius_mean, data.texture_mean, data.perimeter_mean, data.area_mean, data.smoothness_mean,
            data.compactness_mean, data.concavity_mean, data.concave_points_mean, data.symmetry_mean, data.fractal_dimension_mean,
            
            # 2. Standard Error Metrics (10 features)
            data.radius_se, data.texture_se, data.perimeter_se, data.area_se, data.smoothness_se,
            data.compactness_se, data.concavity_se, data.concave_points_se, data.symmetry_se, data.fractal_dimension_se,
            
            # 3. Worst Metrics (10 features)
            data.radius_worst, data.texture_worst, data.perimeter_worst, data.area_worst, data.smoothness_worst,
            data.compactness_worst, data.concavity_worst, data.concave_points_worst, data.symmetry_worst, data.fractal_dimension_worst
        ]
        
        # Verify feature count before passing it to the pipeline
        if len(input_features) != 30:
            raise ValueError(f"Feature list count misaligned! Expected 30, got {len(input_features)}")
        
        # Reshape into a 2D matrix (1 row, 30 columns)
        raw_array = np.array([input_features])
        
        # Scale the features using the production scaler
        scaled_features = scaler.transform(raw_array)
        
        # Run live model inference
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0][prediction])
        
        return {
            "prediction_code": prediction,
            "prediction": "Malignant" if prediction == 1 else "Benign",
            "confidence": round(probability * 100, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")