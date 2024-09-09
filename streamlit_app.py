import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load the saved model using joblib
model = joblib.load('model.joblib')

# Load the image at the top of the app
image = Image.open('./image.png')

# Display the image
st.image(image, use_column_width=True)

# Title of the app
st.title("Medical Insurance Charges Prediction App")

# Create input fields for the features
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ['male', 'female'])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])

# Preprocess inputs for the model
def preprocess_inputs(age, sex, bmi, children, smoker, region):
    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'male' else 0
    smoker = 1 if smoker == 'yes' else 0

    # Map regions to numerical values
    region_dict = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    region = region_dict[region]

    # Return the processed input data as a numpy array
    return np.array([[age, sex, bmi, children, smoker, region]])

# Button to trigger the prediction
if st.button('Predict'):
    # Preprocess inputs
    inputs = preprocess_inputs(age, sex, bmi, children, smoker, region)

    # Make prediction
    prediction = model.predict(inputs)

    # Display the prediction result
    st.success(f"The predicted insurance charge is: ${prediction[0]:,.2f}")
