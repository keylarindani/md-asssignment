import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic 2.csv")

# Preprocessing
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=['NObeyesdad'])  # Target label
y = data['NObeyesdad']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Prediksi Obesitas dengan Random Forest")
st.info("Aplikasi ini menggunakan Machine Learning untuk memprediksi obesitas")

# Menampilkan data
if st.checkbox("Tampilkan Raw Data"):
    st.write(data.head())

# Input Data User
st.sidebar.header("Masukkan Data")
input_data = {}
for col in X.columns:
    if data[col].dtype == 'float64' or data[col].dtype == 'int64':
        input_data[col] = st.sidebar.slider(f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    else:
        options = list(label_encoders[col].classes_)
        input_data[col] = label_encoders[col].transform([st.sidebar.selectbox(f"{col}", options)])[0]

# Menampilkan inputan user
st.write("### Data yang Anda Masukkan")
st.write(pd.DataFrame([input_data]))

# Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediksi = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)
    st.write(f"### Hasil Prediksi: {prediksi}")
    st.write("### Probabilitas Tiap Kelas:")
    st.write(probabilitas)
