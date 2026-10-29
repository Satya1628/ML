
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "best_model.joblib")
model = bundle["model"]
st.set_page_config(page_title="Flight Fare Estimator", page_icon="✈️", layout="centered")
st.title("✈️ Airline Ticket Price Estimator")
st.write("Estimate a historical-style fare from flight details. This is an ML estimate, not a live airline quote.")
st.caption(f"Model: {bundle['model_name']} | Test R²: {bundle['test_metrics']['R2']:.3f}")

with st.form("flight_details"):
    c1, c2 = st.columns(2)
    airline = c1.selectbox("Airline", ["AirAsia", "Air India", "GO FIRST", "IndiGo", "SpiceJet", "Vistara"])
    source = c2.selectbox("Source city", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
    destination_options = [x for x in ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"] if x != source]
    destination = c1.selectbox("Destination city", destination_options)
    departure = c2.selectbox("Departure time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    arrival = c1.selectbox("Arrival time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    stops = c2.selectbox("Stops", ["zero", "one", "two_or_more"])
    cabin = c1.selectbox("Class", ["Economy", "Business"])
    duration = c2.number_input("Flight duration (hours)", min_value=0.5, max_value=60.0, value=2.5, step=0.25)
    days_left = c1.slider("Days before departure", min_value=1, max_value=49, value=15)
    submitted = st.form_submit_button("Estimate fare", type="primary")

if submitted:
    row = pd.DataFrame([{
        "airline": airline, "source_city": source, "departure_time": departure,
        "stops": stops, "arrival_time": arrival, "destination_city": destination,
        "class": cabin, "duration": duration, "days_left": days_left
    }])
    price = float(model.predict(row)[0])
    st.metric("Estimated ticket price", f"₹{price:,.0f}")
    st.info("Treat this as an approximate estimate based on the training data. Actual fares change with availability, timing, promotions and airline policies.")
