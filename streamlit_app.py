import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Judul aplikasi
st.title('📊 Obesity Dataset Explorer')

# 1. Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic 2.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

if data is not None:
    # 2. Menampilkan raw data
    st.header('1. Raw Data Preview')
    
    # Tampilkan 100 baris pertama
    st.write("**First 100 rows:**")
    st.dataframe(data.head(100))
    
    # 3. Informasi dasar dataset
    st.header('2. Dataset Information')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", data.shape[0])
        st.write("**Columns:**", ", ".join(data.columns))
        
    with col2:
        st.metric("Total Columns", data.shape[1])
        st.write("**Target Variable:**", "NObeyesdad")
    
    # 4. Statistik numerik
    st.header('3. Numerical Statistics')
    st.dataframe(data.describe())
    
    # 5. Statistik kategorikal
    st.header('4. Categorical Data Summary')
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        st.subheader(f"Column: {col}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Unique values:", data[col].unique())
        with col2:
            st.write("Value counts:")
            st.dataframe(data[col].value_counts())
    
    # 6. Eksplorasi target variable
    st.header('5. Target Variable Analysis (NObeyesdad)')
    st.bar_chart(data['NObeyesdad'].value_counts())
else:
    st.warning("Data tidak dapat dimuat. Pastikan file 'ObesityDataSet_raw_and_data_sinthetic.csv' ada di direktori yang sama.")
