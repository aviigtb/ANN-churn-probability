import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle


model = tf.keras.models.load_model('Model.h5')

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


st.title("Bank Customer Churn Prediction")

#user input
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value = 30)
balance = st.number_input("Balance", min_value=0.0, value = 50000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value = 650)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value = 50000.0)
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value = 3)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value = 2)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

input_data = pd.DataFrame({
    'CreditScore': [credit_score], 
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],  
    'Balance': [balance],   
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

pred = model.predict(input_data_scaled)
pred_prob = pred[0][0]

if pred_prob > 0.5:
    st.write(" The customer is likely to churn with a probability of {:.2f}%".format(pred_prob * 100))
else:
    st.write(" The customer is not likely to churn with a probability of {:.2f}%".format((1 - pred_prob) * 100))
