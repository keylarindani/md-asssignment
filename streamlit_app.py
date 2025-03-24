import streamlit as st
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic 2.csv")
    return data

data = load_data()

# UI Header
st.title("Machine Learning App")
st.info("This app will predict your obesity level!")

# Menampilkan Raw Data
st.subheader("Data")
if st.checkbox("Tampilkan Raw Data"):
    st.write("This is a raw data")
    st.dataframe(data.head(10))  # Tampilkan 10 data pertama
