import streamlit as st
import pickle
import numpy as np

# Load Model and Scaler
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Credit Card Default Prediction")

st.write("Enter customer details:")

LIMIT_BAL = st.number_input("Credit Limit")
SEX = st.selectbox("Sex", [1,2])
EDUCATION = st.selectbox("Education",[1,2,3,4])
MARRIAGE = st.selectbox("Marriage",[1,2,3])
AGE = st.number_input("Age")

PAY_0 = st.number_input("PAY_0")
PAY_2 = st.number_input("PAY_2")
PAY_3 = st.number_input("PAY_3")
PAY_4 = st.number_input("PAY_4")
PAY_5 = st.number_input("PAY_5")
PAY_6 = st.number_input("PAY_6")

BILL_AMT1 = st.number_input("BILL_AMT1")
BILL_AMT2 = st.number_input("BILL_AMT2")
BILL_AMT3 = st.number_input("BILL_AMT3")
BILL_AMT4 = st.number_input("BILL_AMT4")
BILL_AMT5 = st.number_input("BILL_AMT5")
BILL_AMT6 = st.number_input("BILL_AMT6")

PAY_AMT1 = st.number_input("PAY_AMT1")
PAY_AMT2 = st.number_input("PAY_AMT2")
PAY_AMT3 = st.number_input("PAY_AMT3")
PAY_AMT4 = st.number_input("PAY_AMT4")
PAY_AMT5 = st.number_input("PAY_AMT5")
PAY_AMT6 = st.number_input("PAY_AMT6")

if st.button("Predict"):

    input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                            BILL_AMT1, BILL_AMT2, BILL_AMT3,
                            BILL_AMT4, BILL_AMT5, BILL_AMT6,
                            PAY_AMT1, PAY_AMT2, PAY_AMT3,
                            PAY_AMT4, PAY_AMT5, PAY_AMT6]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer may DEFAULT payment")
    else:
        st.success("Customer will NOT default")
