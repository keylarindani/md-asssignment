# obesity_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ObesityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.features = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC']
        self.target = 'Obesity_Level'
        self.is_trained = False
        
    def preprocess_data(self, data):
        """Handle encoding and normalization"""
        df = data.copy()
        
        # Encode categorical features
        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col])
            df[col] = self.label_encoders[col].transform(df[col])
        
        # Normalize numerical features
        numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP']
        if hasattr(self, 'scaler'):
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def train(self, data):
        """Train the Random Forest model"""
        X = data[self.features]
        y = data[self.target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess
        X_train = self.preprocess_data(X_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        X_test_processed = self.preprocess_data(X_test)
        y_pred = self.model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return accuracy
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if not self.is_trained:
            raise Exception("Model not trained yet. Call train() first.")
            
        input_df = pd.DataFrame([input_data])
        processed_data = self.preprocess_data(input_df)
        prediction = self.model.predict(processed_data)
        return prediction[0]

# app.py
import streamlit as st
import pandas as pd
from obesity_predictor import ObesityPredictor

# Sample data (in a real app, you would load this from a file)
sample_data = pd.DataFrame({
    'Gender': ['Female', 'Female', 'Male', 'Male', 'Male', 'Male', 'Female', 'Male', 'Male', 'Male'],
    'Age': [21, 21, 23, 27, 22, 29, 23, 22, 24, 22],
    'Height': [1.62, 1.52, 1.8, 1.8, 1.78, 1.62, 1.5, 1.64, 1.78, 1.72],
    'Weight': [64, 56, 77, 87, 89.8, 53, 55, 53, 64, 68],
    'family_history_with_overweight': ['yes', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'no', 'yes', 'yes'],
    'FAVC': ['no', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes'],
    'FCVC': [2, 3, 2, 3, 2, 2, 3, 2, 3, 2],
    'NCP': [3, 3, 3, 3, 1, 3, 3, 3, 3, 3],
    'CAEC': ['Sometimes', 'Sometimes', 'Sometimes', 'Sometimes', 'Sometimes', 'Sometimes', 'Sometimes', 'Sometimes', 'Sometimes', 'Sometimes'],
    'Obesity_Level': ['Normal_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 
                     'Obesity_Type_I', 'Insufficient_Weight', 'Normal_Weight', 'Insufficient_Weight',
                     'Overweight_Level_I', 'Overweight_Level_I']
})

# Initialize predictor
predictor = ObesityPredictor()

# Streamlit app
st.title("Machine Learning App")
st.write("This app will predict your obesity level!")
st.write("---")

# Display raw data
st.subheader("Data")
st.write("This is a raw data")
st.dataframe(sample_data)

# Train model
if st.button("Train Model"):
    accuracy = predictor.train(sample_data)
    st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")

# Prediction form
st.subheader("Make Prediction")
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    family_history = st.selectbox("Family History with Overweight", ["yes", "no"])
    favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", ["yes", "no"])
    fcvc = st.number_input("Frequency of consumption of vegetables (FCVC)", min_value=1, max_value=3, value=2)
    ncp = st.number_input("Number of main meals (NCP)", min_value=1, max_value=4, value=3)
    caec = st.selectbox("Consumption of food between meals (CAEC)", ["Sometimes", "Frequently", "Always", "no"])
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = {
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history_with_overweight': family_history,
            'FAVC': favc,
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': caec
        }
        
        try:
            prediction = predictor.predict(input_data)
            st.success(f"Predicted Obesity Level: {prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
