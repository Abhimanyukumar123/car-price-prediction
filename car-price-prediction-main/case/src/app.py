import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "used_cars.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Drop unwanted column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🚗 Used Car Price Prediction")
st.subheader("Estimate resale value instantly")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🔧 Car Details")

year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=2000,
    max_value=2025,
    value=2020
)

km = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000
)

brand = st.sidebar.selectbox(
    "Brand",
    sorted(df['brand'].dropna().astype(str).unique())
)

model_name = st.sidebar.selectbox(
    "Model",
    sorted(
        df[df['brand'] == brand]['model']
        .dropna()
        .astype(str)
        .unique()
    )
)

fuel = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df['fuel_type'].dropna().astype(str).unique())
)

transmission = st.sidebar.selectbox(
    "Transmission",
    sorted(df['transmission_type'].dropna().astype(str).unique())
)

city = st.sidebar.selectbox(
    "City",
    sorted(df['city'].dropna().astype(str).unique())
)

owners = st.sidebar.selectbox(
    "Number of Owners",
    sorted(df['number_of_owners'].dropna().astype(int).unique())
)

# --------------------------------------------------
# AUTO BODY TYPE
# --------------------------------------------------
body_series = df[
    (df['brand'] == brand) &
    (df['model'] == model_name)
]['bodytype']

bodytype = body_series.mode()[0] if not body_series.empty else "Unknown"

# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------
st.markdown("### 📋 Car Summary")

st.markdown(f"""
- **Brand:** {brand}
- **Model:** {model_name}
- **Year:** {year}
- **Kilometers Driven:** {km:,} km
- **Fuel Type:** {fuel}
- **Transmission:** {transmission}
- **City:** {city}
- **Owners:** {owners}
- **Body Type:** {bodytype}
""")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.markdown("### 💰 Price Prediction")

if st.button("🚀 Predict Price"):
    payload = {
        "manufacturing_year": int(year),
        "km_driven": int(km),
        "brand": brand,
        "model": model_name,
        "fuel_type": fuel,
        "transmission_type": transmission,
        "city": city,
        "bodytype": bodytype,
        "number_of_owners": int(owners)
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"💵 Estimated Price: ₹ {price:,}")
        else:
            st.error("❌ Prediction failed. Check API.")

    except Exception as e:
        st.error(f"❌ API Error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Developed by <b>Team Plan B</b> | ML Project</center>",
    unsafe_allow_html=True
)
