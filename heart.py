import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model from pickle
with open('heart_failure.pkl', 'rb') as file:
    model_pkl = pickle.load(file)

# Load scaler from pickle
with open('scaler.pkl', 'rb') as file:
    scaler_pkl = pickle.load(file)

# Function to preprocess input data
def preprocess_data(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    return np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

# Function to make prediction
def make_prediction(input_data):
    # Standardize input data
    input_data_scaled = scaler_pkl.transform(input_data)
    # Predict with the trained model
    prediction = model_pkl.predict(input_data_scaled)
    prediction_proba = model_pkl.predict_proba(input_data_scaled)
    return prediction, prediction_proba

# Create input widgets using Streamlit
st.title('Heart Disease Prediction')
st.write('Please enter the following information:')
# Additional text input fields for selected numeric values
custom_values = st.checkbox('Input Custom Numeric Values')


# Define layout using columns
col1, col2, col3 = st.columns([1, 1, 1])

# Input column - Numerical Inputs
with col1:
    if custom_values:
        age = st.number_input('Age', min_value=0, max_value=120)
        resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200)
        cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=600)
        max_hr = st.number_input('Max Heart Rate', min_value=60, max_value=220)
        oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0)
    else:
        age = st.slider('Age', 0, 120, 50)
        resting_bp = st.slider('Resting Blood Pressure (mm Hg)', 50, 200, 120)
        cholesterol = st.slider('Cholesterol (mg/dL)', 100, 600, 200)
        max_hr = st.slider('Max Heart Rate', 60, 220, 150)
        oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 10.0, 3.0)

# Input column - Categorical Inputs
with col2:
    sex = st.radio('Sex', ['Male', 'Female'])
    chest_pain_type = st.selectbox('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])
    fasting_bs = st.selectbox('Fasting Blood Sugar', ['0', '1'], format_func=lambda x: 'No' if x == '0' else 'Yes')
    resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'], format_func=lambda x: x)
    exercise_angina = st.radio('Exercise Induced Angina', ['No', 'Yes'])
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'], format_func=lambda x: x)

# Prediction column
with col3:
    if st.button('Predict'):
        # Convert inputs to appropriate data types
        if custom_values:
            age = int(age) if age else 50
            resting_bp = float(resting_bp) if resting_bp else 120
            cholesterol = float(cholesterol) if cholesterol else 200
            max_hr = float(max_hr) if max_hr else 150
            oldpeak = float(oldpeak) if oldpeak else 3.0
        else:
            age = int(age)
            resting_bp = float(resting_bp)
            cholesterol = float(cholesterol)
            max_hr = float(max_hr)
            oldpeak = float(oldpeak)

        # Convert categorical inputs to numerical
        sex = 1 if sex == 'Male' else 0
        chest_pain_mapping = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
        resting_ecg_mapping = {'Normal': 0, 'ST': 1, 'LVH': 2}
        fasting_bs = int(fasting_bs)
        exercise_angina = 1 if exercise_angina == 'Yes' else 0
        st_slope_mapping = {'Up': 1, 'Flat': 0, 'Down': -1}

        chest_pain_type = chest_pain_mapping[chest_pain_type]
        resting_ecg = resting_ecg_mapping[resting_ecg]
        st_slope = st_slope_mapping[st_slope]

        input_data = preprocess_data(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
        prediction, prediction_proba = make_prediction(input_data)
        
        if prediction[0]:
            prediction_result = "**Heart Disease: Yes**"
            probability_yes = f"Probability (Yes): {prediction_proba[0][1]*100:.2f}%"
            probability_no = f"Probability (No): {prediction_proba[0][0]*100:.2f}%"
            st.warning("Please be aware! Heart failure is indicated.")
        else:
            prediction_result = "**Heart Disease: No**"
            probability_yes = f"Probability (Yes): {prediction_proba[0][1]*100:.2f}%"
            probability_no = f"Probability (No): {prediction_proba[0][0]*100:.2f}%"
            st.write("Keep up the good work! Maintain a healthy lifestyle to prevent heart diseases.")

        st.write(prediction_result)
        st.write(probability_yes)
        st.write(probability_no)

        # Plotting probability
        labels = ['Heart Disease No', 'Heart Disease Yes']
        values = prediction_proba[0]
        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['blue', 'red'])
        ax.set_ylabel('Probability')
        ax.set_title('Probability of Heart Disease')
        st.pyplot(fig)
