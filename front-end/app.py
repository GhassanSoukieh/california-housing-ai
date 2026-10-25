
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="California Housing Predictor",
    layout="centered"
)

st.title("California Housing Predictor")
st.write("Enter housing information to estimate the property value using AI.")

# Backend connection
with st.sidebar:
    st.header("Backend Status")

    if st.button("Test Backend"):
        try:
            response = requests.get(f"{API_URL}/", timeout=5)
            response.raise_for_status()
            st.success(response.json()["message"])
        except requests.exceptions.RequestException:
            st.error("Cannot connect to the backend!")

# Housing inputs
st.subheader("Property Information")

with st.form("housing_form"):

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Median Income",
            min_value=0.0,
            value=4.445,
            help="Median household income in units of $10,000"
        )

        house_age = st.number_input(
            "House Age",
            min_value=0.0,
            value=52.0
        )

        ave_rooms = st.number_input(
            "Average Rooms",
            min_value=0.0,
            value=5.5346
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms",
            min_value=0.0,
            value=1.1509
        )

    with col2:
        population = st.number_input(
            "Population",
            min_value=0.0,
            value=742.0
        )

        ave_occup = st.number_input(
            "Average Occupancy",
            min_value=0.0,
            value=2.3333
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=37.77
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-122.43
        )

    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary",
        use_container_width=True
    )

# Prediction request
if submitted:

    housing_data = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }

    try:
        with st.spinner("Predicting and generating AI explanation..."):
            response = requests.post(
                f"{API_URL}/prediction_explanation",
                json=housing_data,
                timeout=180
            )
            response.raise_for_status()
            result = response.json()

        price = result["price_usd"]

        st.success("Prediction completed!")
        st.metric(
            label="Estimated Median Housing Value",
            value=f"${price:,.2f}"
        )

        if "model" in result:
            st.caption(f"Prediction model: {result['model']}")

        st.divider()
        st.subheader("AI Explanation")
        st.write(result["explanation"])

        with st.expander("View Input Data"):
            st.json(housing_data)

    except requests.exceptions.RequestException as e:
        st.error(f"Prediction request failed: {e}")

    except (KeyError, TypeError, ValueError):
        st.error("The backend returned an unexpected response.")
