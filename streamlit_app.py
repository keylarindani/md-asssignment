import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from obesity_model import ObesityPredictor

# Initialize model
@st.cache_resource
def load_model():
    return ObesityPredictor('data/ObesityDataSet.csv')

model = load_model()

# Streamlit app
st.title('Obesity Level Prediction')

# 1. Show raw data
st.header('1. Raw Data')
if st.checkbox('Show raw data'):
    st.write(model.data)

# 2. Data Visualization
st.header('2. Data Visualization')
plot_type = st.selectbox('Select plot type', 
                        ['Count Plot by Gender', 
                         'Age Distribution', 
                         'Height vs Weight',
                         'Obesity Level Distribution'])

fig, ax = plt.subplots()
if plot_type == 'Count Plot by Gender':
    sns.countplot(data=model.data, x='Gender', ax=ax)
elif plot_type == 'Age Distribution':
    sns.histplot(data=model.data, x='Age', kde=True, ax=ax)
elif plot_type == 'Height vs Weight':
    sns.scatterplot(data=model.data, x='Height', y='Weight', hue='Gender', ax=ax)
elif plot_type == 'Obesity Level Distribution':
    sns.countplot(data=model.data, x='NObeyesdad', ax=ax)
    plt.xticks(rotation=45)

st.pyplot(fig)

# 3. User input for prediction
st.header('3. Make a Prediction')

# Numerical inputs
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider('Age', min_value=10, max_value=100, value=25)
with col2:
    height = st.slider('Height (m)', min_value=1.0, max_value=2.5, value=1.7, step=0.01)
with col3:
    weight = st.slider('Weight (kg)', min_value=30, max_value=200, value=70)

# Categorical inputs
col4, col5, col6 = st.columns(3)
with col4:
    gender = st.selectbox('Gender', ['Male', 'Female'])
with col5:
    family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
with col6:
    favc = st.selectbox('Frequent consumption of high caloric food', ['yes', 'no'])

col7, col8, col9 = st.columns(3)
with col7:
    fcvc = st.slider('Frequency of vegetables consumption', 1, 3, 2)
with col8:
    ncp = st.slider('Number of main meals', 1, 4, 3)
with col9:
    caec = st.selectbox('Consumption of food between meals', 
                       ['no', 'Sometimes', 'Frequently', 'Always'])

# Create input dataframe
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
    # Add other features with default values
    'SMOKE': ['no'],
    'CH2O': [2],
    'SCC': ['no'],
    'FAF': [0],
    'TUE': [0],
    'CALC': ['no'],
    'MTRANS': ['Public_Transportation']
})

# 5. Show user input
st.header('4. Your Input Data')
st.write(input_data)

# Make prediction
if st.button('Predict Obesity Level'):
    probabilities, prediction = model.predict(input_data)
    
    # 6. Show probabilities
    st.header('5. Prediction Probabilities')
    prob_df = pd.DataFrame({
        'Class': model.label_encoder.classes_,
        'Probability': probabilities
    })
    st.bar_chart(prob_df.set_index('Class'))
    
    # 7. Show final prediction
    st.header('6. Final Prediction')
    st.success(f'The predicted obesity level is: {prediction}')
