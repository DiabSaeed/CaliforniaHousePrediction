import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="California House Price Predictor")

PREPROCESSOR_PATH = "preprocessor.joblib"
MODEL_PATH = "xgb_model.joblib"


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def make_input_df(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity,
):
    return pd.DataFrame(
        {
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity],
        }
    )


st.title(" California House Price Predictor")
st.write("Enter Pricing Data")

try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error(f"Erro during loading files  : {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23, format="%.5f")
    latitude = st.number_input("Latitude", value=37.88, format="%.5f")
    housing_median_age = st.number_input("Housing median age", min_value=1, value=20)
    total_rooms = st.number_input("Total rooms", min_value=1, value=2000)
    total_bedrooms = st.number_input("Total bedrooms", min_value=1, value=400)

with col2:
    population = st.number_input("Population", min_value=1, value=1000)
    households = st.number_input("Households", min_value=1, value=350)
    median_income = st.number_input("Median income", min_value=0.0, value=5.0, format="%.2f")
    ocean_proximity = st.selectbox(
        "Ocean proximity",
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
    )

if st.button("Predict"):
    input_df = make_input_df(
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        ocean_proximity,
    )

    transformed = preprocessor.transform(input_df)
    pred_log = model.predict(transformed)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"Predicted price: ${pred_price:,.2f}")

    with st.expander("Show input data"):
        st.dataframe(input_df)