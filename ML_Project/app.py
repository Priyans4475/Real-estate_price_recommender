import streamlit as st
import pandas as pd
import pickle
import numpy as np

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPE_PATH = os.path.join(BASE_DIR, 'pipelines.pkl')
DF_PATH = os.path.join(BASE_DIR, 'df.pkl')

with open(PIPE_PATH, 'rb') as f:
    pipe = pickle.load(f)

with open(DF_PATH, 'rb') as f:
    X = pickle.load(f)


st.title("House Price Prediction App")

# ---- Inputs ----
property_type = st.selectbox(
    "Property Type",
    sorted(X['property_type'].unique())
)

sector = st.selectbox(
    "Sector",
    sorted(X['sector'].unique().tolist())
)

bedRoom = st.number_input("Bedrooms", 1, 10, 4)
bathroom = st.number_input("Bathrooms", 1, 10, 3)

balcony = st.selectbox(
    "Balcony",
    sorted(X['balcony'].unique())
)

agePossession = st.selectbox(
    "Age / Possession",
    sorted(X['agePossession'].unique())
)

built_up_area = st.number_input("Built-up Area (sqft)", value=2750)

servant_room = st.selectbox(
    "Servant Room",
    sorted(X['servant room'].unique())
)

store_room = st.selectbox(
    "Store Room",
    sorted(X['store room'].unique())
)

furnishing_type = st.selectbox(
    "Furnishing Type",
    sorted(X['furnishing_type'].unique())
)

luxury_category = st.selectbox(
    "Luxury Category",
    sorted(X['luxury_category'].unique())
)

floor_category = st.selectbox(
    "Floor Category",
    sorted(X['floor_category'].unique())
)

# ---- Prediction ----
if st.button("Predict Price"):
    input_df = pd.DataFrame([[
        property_type, sector, bedRoom, bathroom, balcony,
        agePossession, built_up_area, servant_room, store_room,
        furnishing_type, luxury_category, floor_category
    ]], columns=[
        'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
        'agePossession', 'built_up_area', 'servant room', 'store room',
        'furnishing_type', 'luxury_category', 'floor_category'
    ])

    log_price = pipe.predict(input_df)[0]
    price = np.expm1(log_price)   # 🔥 IMPORTANT

    st.success(f"💰 Predicted Price: ₹ {price:,.2f} Cr")

