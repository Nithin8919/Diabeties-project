#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:35:31 2024

@author: nitin
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('/Users/nitin/Documents/diabeties project/trained_model.sav', 'rb'))

# creating a function for Prediction
def diabetes_prediction(input_data):
    # convert input data to float
    input_data = [float(x) for x in input_data]
    # reshape the array as we are predicting for one instance
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    # Display prediction result
    st.write(diagnosis)

if __name__ == "__main__":
    main()
