import streamlit as st
import pandas as pd
import joblib
from src.preprocessing import clean_text

model = joblib.load("models/model.pkl")

st.set_page_config(page_title="Clinical NLP Dashboard", layout="wide")

st.title("🏥 Clinical NLP Classification System")

user_input = st.text_area("Enter Clinical Text")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    prediction = model.predict([cleaned])[0]

    st.success(f"Predicted Specialty: {prediction}")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['clean_text'] = df['text'].apply(clean_text)
    df['prediction'] = model.predict(df['clean_text'])

    st.dataframe(df.head())

    st.bar_chart(df['prediction'].value_counts())