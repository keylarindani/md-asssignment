pip install streamlit pandas matplotlib seaborn
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.dataframe(data.head(10))  # Menampilkan 10 data pertama

fig, ax = plt.subplots(figsize=(8, 5))
for category in data["NObeyesdad"].unique():
    subset = data[data["NObeyesdad"] == category]
    ax.scatter(subset["Height"], subset["Weight"], label=category, alpha=0.7)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter Plot: Height vs Weight")
plt.legend()
plt.grid(True)
st.pyplot(fig)

