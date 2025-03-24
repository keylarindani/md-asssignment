import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic 2.csv")  # Sesuaikan dengan file dataset kamu
    return data

data = load_data()

# Menampilkan raw data
st.title("Machine Learning App")
st.info("This app will predict your obesity level!")

with st.expander("Data"):
    st.write("This is a raw data")
    st.dataframe(data)

# Scatter Plot
st.subheader("Data Visualization")

fig, ax = plt.subplots(figsize=(8, 5))
scatter = sns.scatterplot(
    data=data,
    x="Height",
    y="Weight",
    hue="NObeyesdad",
    palette="rainbow",   # Warna kategori
    alpha=0.8,           # Transparansi titik
    edgecolor="black",   # Outline hitam agar lebih jelas
    s=50                 # Ukuran titik
)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter Plot: Height vs Weight")
plt.legend(title="NObeyesdad", bbox_to_anchor=(1.05, 1), loc='upper left')  # Pindahkan legend ke kanan
plt.grid(True)

# Menampilkan plot di Streamlit
st.pyplot(fig)
