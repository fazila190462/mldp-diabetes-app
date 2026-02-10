import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# ======================
# Load Model from Google Drive
# ======================

MODEL_PATH = "diabetes_rf_pipeline.joblib"
FILE_ID = "1W6tTSxi9_tqnhH3ausQ6W16J_YfsBhp3"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

# ======================
# UI
# ======================

st.title("🩺 Diabetes Risk Prediction")
st.write("Enter basic health indicators to estimate diabetes risk.")

# ======================
# Age Mapping
# ======================

age_map = {
    "18-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69": 10,
    "70-74": 11,
    "75-79": 12,
    "80+": 13
}

# ======================
# USER INPUTS (ONLY 6)
# ======================

HighBP = st.selectbox("High Blood Pressure", [0,1])
HighChol = st.selectbox("High Cholesterol", [0,1])
BMI = st.slider("BMI", 10, 50, 25)
PhysActivity = st.selectbox("Physical Activity (Past 30 days)", [0,1])
GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

age_label = st.selectbox("Age Group", list(age_map.keys()))
Age = age_map[age_label]

# ======================
# Prediction
# ======================

if st.button("Predict Diabetes Risk"):

    # Hidden default values for remaining features
    input_data = {
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": 1,
        "BMI": BMI,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": PhysActivity,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": GenHlth,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": Age,
        "Education": 4,
        "Income": 5
    }

    df_input = pd.DataFrame([input_data])

    # Align with training features
    if hasattr(model, "feature_names_in_"):
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Diabetes Risk (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Diabetes Risk (Probability: {probability:.2%})")
