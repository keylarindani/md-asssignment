import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('ObesityDataSet_raw_and_data_sinthetic 2.csv')

data = load_data()

# Preprocessing
def preprocess_data(df):
    # Pisahkan fitur dan target
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    
    # Identifikasi kolom
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Buat preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    return preprocessor, X, y

preprocessor, X, y = preprocess_data(data)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(preprocessor.transform(X), y_encoded)
    return model

model = train_model()

# Streamlit App
st.title('Obesity Level Prediction')

# 1. Tampilkan raw data
st.header('1. Raw Data')
if st.checkbox('Show raw data'):
    st.write(data)

# 2. Visualisasi data
st.header('2. Data Visualization')
plot_options = ['Gender Distribution', 'Age Distribution', 
               'Height vs Weight', 'Obesity Level Distribution']
selected_plot = st.selectbox('Choose a plot:', plot_options)

fig, ax = plt.subplots()
if selected_plot == 'Gender Distribution':
    sns.countplot(data=data, x='Gender', ax=ax)
elif selected_plot == 'Age Distribution':
    sns.histplot(data=data, x='Age', bins=20, kde=True, ax=ax)
elif selected_plot == 'Height vs Weight':
    sns.scatterplot(data=data, x='Height', y='Weight', hue='Gender', ax=ax)
elif selected_plot == 'Obesity Level Distribution':
    sns.countplot(data=data, x='NObeyesdad', ax=ax)
    plt.xticks(rotation=45)

st.pyplot(fig)

# 3. Input data dari user
st.header('3. Input Data for Prediction')

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Gender', ['Female', 'Male'])
    age = st.slider('Age', 14, 80, 25)
    height = st.slider('Height (m)', 1.40, 2.20, 1.70)
    weight = st.slider('Weight (kg)', 30, 200, 70)
    
with col2:
    family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
    favc = st.selectbox('Frequent high caloric food consumption', ['yes', 'no'])
    fcvc = st.slider('Vegetable consumption frequency (1-3)', 1, 3, 2)
    ncp = st.slider('Number of main meals (1-4)', 1, 4, 3)

caec = st.selectbox('Eating between meals', 
                   ['no', 'Sometimes', 'Frequently', 'Always'])

# 4. Tampilkan input user
st.header('4. Your Input Data')
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'family_history_with_overweight': [family_history],
    'FAVC': [favc],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'CAEC': [caec],
    # Default values for other features
    'SMOKE': ['no'],
    'CH2O': [2],
    'SCC': ['no'],
    'FAF': [0],
    'TUE': [0],
    'CALC': ['no'],
    'MTRANS': ['Public_Transportation']
})

st.write(input_data)

# 5. Prediksi
if st.button('Predict Obesity Level'):
    # Preprocess input
    input_processed = preprocessor.transform(input_data)
    
    # Prediksi
    probabilities = model.predict_proba(input_processed)[0]
    prediction = model.predict(input_processed)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    
    # 6. Tampilkan probabilities
    st.header('5. Prediction Probabilities')
    prob_df = pd.DataFrame({
        'Obesity Level': label_encoder.classes_,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_df.set_index('Obesity Level'))
    
    # 7. Tampilkan hasil prediksi
    st.header('6. Final Prediction')
    st.success(f'Predicted Obesity Level: {prediction_label}')
