import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
import numpy as np

# Function to load the model and preprocessor
@st.cache_data
def load_model():
    model = joblib.load('knn_model.pkl') 
    preprocessor = joblib.load('preprocessor.pkl') 
    return model, preprocessor

# Load the model and preprocessor
model, preprocessor = load_model()

# Function to preprocess input data
def preprocess_input(age, sex, cp, trestbps, chol, fbs, recg, mhr, ea, op, sts, preprocessor):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [cp],
        'RestingBP': [trestbps],
        'Cholesterol': [chol],
        'FastingBS': [fbs],
        'RestingECG': [recg],
        'MaxHR': [mhr],
        'ExerciseAngina': [ea],
        'Oldpeak': [op],
        'ST_Slope': [sts]
    })

    # Convert categorical variables to appropriate format
    input_data['Sex'] = 1 if sex == 'male' else 0
    input_data['FastingBS'] = 1 if fbs == 'true' else 0
    input_data['RestingECG'] = 1 if recg == 'ST' else 0
    input_data['ExerciseAngina'] = 1 if ea == 'yes' else 0
    input_data['ST_Slope'] = 1 if sts == 'flat' else 0

    # Apply preprocessing pipeline
    input_data_preprocessed = preprocessor.transform(input_data)

    return input_data_preprocessed, input_data

# Streamlit app main function
def main():
    st.title('Heart Failure Prediction')
    st.sidebar.header('Input Features')

    # Define input widgets
    age = st.sidebar.slider('Age', 20, 80, 40)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    cp = st.sidebar.slider('Chest Pain Type', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol Level', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['true', 'false'])
    recg = st.sidebar.selectbox('Resting ECG', ['normal', 'ST'])
    mhr = st.sidebar.slider('Max HR', 90, 200, 120)
    ea = st.sidebar.selectbox('Exercise Angina', ['no', 'yes'])
    op = st.sidebar.slider('Oldpeak', -2.0, 6.0, 1.0, step=0.1)
    sts = st.sidebar.selectbox('ST_Slope', ['up', 'flat', 'down'])

    # Preprocess input data
    input_data_preprocessed, input_data_raw = preprocess_input(age, sex, cp, trestbps, chol, fbs, recg, mhr, ea, op, sts, preprocessor)

    # Make prediction
    prediction = model.predict(input_data_preprocessed)
    prediction_proba = model.predict_proba(input_data_preprocessed)

    # Display user input data
    st.subheader('Input Data Details')
    st.write(input_data_raw)

    # Display prediction results
    st.subheader('Prediction')
    prediction_text = 'likely' if prediction[0] == 1 else 'not likely'
    st.write(f'The patient is {prediction_text} to have a heart failure.')

    # Display prediction probability
    st.subheader('Prediction Probability')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['No Heart Failure', 'Heart Failure'], y=prediction_proba[0], marker_color=['green', 'red']))
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig)

    # Health tips based on prediction
    st.subheader('Health Tips')
    if prediction[0] == 1:
        st.write("""
            - Regular check-ups with your healthcare provider.
            - Maintain a healthy diet and regular exercise.
            - Monitor and control blood pressure and cholesterol levels.
            - Avoid smoking and limit alcohol intake.
            - Manage stress effectively.
        """)
    else:
        st.write("""
            - Continue to maintain a healthy lifestyle.
            - Regular check-ups to monitor heart health.
            - Stay informed about heart disease prevention.
        """)

if __name__ == '__main__':
    main()
