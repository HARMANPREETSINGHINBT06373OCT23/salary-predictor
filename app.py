import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

# Load trained model and expected columns
model = pickle.load(open('model.sav', 'rb'))
model_columns = pickle.load(open('model_columns.sav', 'rb'))  # List of feature columns used in training

st.title('Player Salary Prediction')
st.sidebar.header('Player Data')

# Load and show image (optional)
try:
    image = Image.open('bb.jpg')
    st.image(image, '')
except FileNotFoundError:
    st.warning("Image 'bb.jpg' not found.")

# Function for input sliders
def user_report():
    rating = st.sidebar.slider('Rating', 50, 100, 75)
    jersey = st.sidebar.slider('Jersey', 0, 100, 23)
    team = st.sidebar.selectbox('Team', ['Team_' + str(i) for i in range(1, 31)])
    position = st.sidebar.selectbox('Position', ['Position_' + str(i) for i in range(1, 6)])
    country = st.sidebar.selectbox('Country', ['Country_' + str(i) for i in range(1, 5)])
    draft_year = st.sidebar.slider('Draft Year', 2000, 2020, 2010)
    draft_round = st.sidebar.slider('Draft Round', 1, 10, 1)
    draft_peak = st.sidebar.slider('Draft Peak', 1, 30, 15)

    # Base numeric data
    data = {
        'rating': rating,
        'jersey': jersey,
        'draft_year': draft_year,
        'draft_round': draft_round,
        'draft_peak': draft_peak
    }

    df = pd.DataFrame([data])

    # One-hot encoded fields (simulate what training data had)
    df[team] = 1
    df[position] = 1
    df[country] = 1

    # Add missing columns as 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]  # Reorder columns to match training
    return df

user_data = user_report()

st.header('Player Data')
st.write(user_data)

# Predict and display salary
salary = model.predict(user_data)
st.subheader('Predicted Salary')
st.subheader(f"${np.round(salary[0], 2):,.2f}")
