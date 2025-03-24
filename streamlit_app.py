import streamlit as st
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic 2.csv")
    return data

data = load_data()

# UI Header
st.markdown("<h1 style='text-align: center;'>Machine Learning App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; background-color: #e3f2fd; padding: 10px; border-radius: 10px;'>This app will predict your obesity level!</p>", unsafe_allow_html=True)

# Menampilkan Raw Data
st.subheader("Data")
if st.checkbox("Tampilkan Raw Data"):
    st.write("This is a raw data")
    st.dataframe(data.head(10))  # Tampilkan 10 data pertama

# Data Visualization
st.subheader("Data Visualization")

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=data, x="Height", y="Weight", hue="NObeyesdad", palette="rainbow", alpha=0.8)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter Plot: Height vs Weight")

st.pyplot(fig)
