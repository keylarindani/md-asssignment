import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Machine Learning App")
st.info("This app will predict your obesity level!")

data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic 2.csv") 
df = pd.DataFrame(data)

# Expander to show raw data
with st.expander("Data"):
    st.write("This is a raw data")
    st.dataframe(df)

#Data Visualilzation
category_mapping = {
    "Insufficient_Weight": "blue",
    "Normal_Weight": "green",
    "Overweight_Level_I": "yellow",
    "Overweight_Level_II": "orange",
    "Obesity_Type_I": "red",
    "Obesity_Type_II": "purple"
}


with st.expander("Data Visualization"):
    fig = px.scatter(df, x="Height", y="Weight", color="NObeyesdad",
                 color_discrete_map=category_mapping,
                 title="Data Visualization",
                 labels={"Height": "Height (m)", "Weight": "Weight (kg)"})

    st.plotly_chart(fig)


def train_model():
    # Dummy data training (harus diganti dengan dataset asli)
    data = pd.DataFrame({
        "Gender": np.random.choice(["Male", "Female"], 100),
        "Age": np.random.randint(10, 80, 100),
        "Height": np.random.uniform(1.2, 2.2, 100),
        "Weight": np.random.randint(30, 200, 100),
        "family_history_with_overweight": np.random.choice(["yes", "no"], 100),
        "FAVC": np.random.choice(["yes", "no"], 100),
        "FCVC": np.random.randint(1, 4, 100),
        "NCP": np.random.randint(1, 5, 100),
        "CAEC": np.random.choice(["Sometimes", "Frequently", "Always", "No"], 100),
        "Obesity": np.random.choice(["Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II", "Obesity Type I", "Obesity Type II"], 100)
    })
    
    # Label encoding
    label_encoders = {}
    for col in ["Gender", "family_history_with_overweight", "FAVC", "CAEC"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    X = data.drop("Obesity", axis=1)
    y = LabelEncoder().fit_transform(data["Obesity"])
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, label_encoders

# Load model
model, encoders = train_model()

# **Judul Aplikasi**
st.title("Obesity Prediction App")

# **Input Data Pengguna**
st.header("Input Data")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 80, 1)
height = st.slider("Height (m)", 1.2, 2.2, 1.2)
weight = st.slider("Weight (kg)", 30, 200, 30)
family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
favc = st.selectbox("Frequent Consumption of High Caloric Food", ["yes", "no"])
fcvc = st.slider("Frequency of Vegetable Consumption", 1, 3, 1)
ncp = st.slider("Number of Main Meals", 1, 4, 1)
caec = st.selectbox("Consumption of Food Between Meals", ["Sometimes", "Frequently", "Always", "No"])

# **Menampilkan Data yang Diinputkan**
input_data = pd.DataFrame([{ 
    "Gender": encoders["Gender"].transform([gender])[0], 
    "Age": age, "Height": height, "Weight": weight,
    "family_history_with_overweight": encoders["family_history_with_overweight"].transform([family_history])[0],
    "FAVC": encoders["FAVC"].transform([favc])[0],
    "FCVC": fcvc, "NCP": ncp, "CAEC": encoders["CAEC"].transform([caec])[0]
}])

st.write("Data input by user")
st.dataframe(input_data)

# **Prediksi Model**
# Mendapatkan probabilitas prediksi untuk setiap kelas
probabilities = model.predict_proba(input_data)

# Mendapatkan probabilitas prediksi untuk setiap kelas
probabilities = model.predict_proba(input_data)

# Menampilkan probabilitas tiap kelas dalam dataframe
st.write("Prediction Probabilities:")
st.dataframe(pd.DataFrame(probabilities, columns=["Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II", "Obesity Type I", "Obesity Type II"]))

# Menampilkan hasil prediksi akhir
predicted_class = np.argmax(probabilities)
st.write("The predicted output is: ", predicted_class)
