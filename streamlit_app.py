import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('ObesityDataSet_raw_and_data_sinthetic 2.csv')

data = load_data()

# Preprocessing
def preprocess_data(df):
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
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

# 1. Show raw data
st.header('1. Raw Data')
if st.checkbox('Show raw data'):
    st.write(data)

# 2. Simple data visualization using Streamlit's native functions
st.header('2. Data Visualization')
viz_option = st.selectbox('Select visualization:', 
                         ['Gender Distribution', 
                          'Age Distribution',
                          'Weight Distribution'])

if viz_option == 'Gender Distribution':
    st.bar_chart(data['Gender'].value_counts())
elif viz_option == 'Age Distribution':
    st.line_chart(data['Age'].value_counts().sort_index())
elif viz_option == 'Weight Distribution':
    st.area_chart(data['Weight'].value_counts().sort_index())

# 3. User input
st.header('3. Input Data for Prediction')

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Gender', ['Female', 'Male'])
    age = st.slider('Age', 14, 80, 25)
    height = st.slider('Height (m)', 1.40, 2.20, 1.70)
    
with col2:
    weight = st.slider('Weight (kg)', 30, 200, 70)
    family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
    favc = st.selectbox('Frequent high caloric food consumption', ['yes', 'no'])

fcvc = st.slider('Vegetable consumption frequency (1-3)', 1, 3, 2)
ncp = st.slider('Number of main meals (1-4)', 1, 4, 3)
caec = st.selectbox('Eating between meals', ['no', 'Sometimes', 'Frequently', 'Always'])

# 4. Show user input
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
    'SMOKE': ['no'],
    'CH2O': [2],
    'SCC': ['no'],
    'FAF': [0],
    'TUE': [0],
    'CALC': ['no'],
    'MTRANS': ['Public_Transportation']
})

st.write(input_data)

# 5. Prediction
if st.button('Predict Obesity Level'):
    input_processed = preprocessor.transform(input_data)
    probabilities = model.predict_proba(input_processed)[0]
    prediction = model.predict(input_processed)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    
    st.header('5. Prediction Probabilities')
    prob_df = pd.DataFrame({
        'Obesity Level': label_encoder.classes_,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_df.set_index('Obesity Level'))
    
    st.header('6. Final Prediction')
    st.success(f'Predicted Obesity Level: {prediction_label}')
