import streamlit as st
import pickle
import numpy as np

# PAGE CONFIG

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

# CUSTOM BACKGROUND

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #dbeafe,
        #fef3c7,
        #dcfce7,
        #fae8ff
    );
}

h1 {
    color: #1e3a8a;
    text-align: center;
}

label {
    color: black !important;
    font-weight: bold;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

input {
    background-color: white !important;
    color: black !important;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL

model = pickle.load(
    open(r"../models/model.pkl", "rb")
)

# LOAD SCALER

scaler = pickle.load(
    open(r"../models/scaler.pkl", "rb")
)

# TITLE

st.title("🏥 Medical Insurance Cost Prediction")

st.write("Enter Patient Details")

# USER INPUTS

age = st.slider(
    "Age",
    18,
    100,
    30
)

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.slider(
    "Number of Children",
    0,
    10,
    1
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    [
        "Southwest",
        "Southeast",
        "Northwest",
        "Northeast"
    ]
)

# MANUAL ENCODING

sex = 1 if sex == "Male" else 0

smoker = 1 if smoker == "Yes" else 0

region_map = {
    "Northeast": 0,
    "Northwest": 1,
    "Southeast": 2,
    "Southwest": 3
}

region = region_map[region]

# PREDICT BUTTON

if st.button("Predict Insurance Cost"):

    features = np.array([
        [
            age,
            sex,
            bmi,
            children,
            smoker,
            region
        ]
    ])

    # SCALE FEATURES

    scaled_features = scaler.transform(
        features
    )

    # PREDICT

    prediction = model.predict(
        scaled_features
    )

    # DISPLAY RESULT

    st.success(
        f"Estimated Medical Insurance Cost: ${prediction[0]:,.2f}"
    )