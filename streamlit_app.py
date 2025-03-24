import streamlit as st
import pandas as pd

# Judul aplikasi
st.title('Obesity Data Analysis')

# 1. Menampilkan raw data
st.header('1. Raw Data')

# Load data
@st.cache_data  # Decorator untuk caching data
def load_data():
    data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    return data

data = load_data()

# Tampilkan data mentah
st.write("Berikut adalah data obesitas yang akan digunakan:")
st.dataframe(data)

# Tampilkan statistik dasar
st.subheader('Statistik Deskriptif Data')
st.write(data.describe())

# Informasi kolom
st.subheader('Informasi Kolom')
st.write(f"Jumlah baris: {data.shape[0]}")
st.write(f"Jumlah kolom: {data.shape[1]}")
st.write("Daftar kolom:")
st.write(list(data.columns))
