import streamlit as st
import joblib
import numpy as np
import pandas as pd

columns=['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN']

st.header("Lung Cancer Prediction App")
st.subheader("Input the required fields and predict the lung cancer ")

name=st.text_input("Enter Your Good Name?",help="e.g. Muhammad Junaid")
gender=st.selectbox("Gender",['Male','Female'],help="Please Enter Your Gender" )
age=st.number_input("Age",min_value=18,max_value=100,help="Please Input Age.Range 18-100")
smoking=st.selectbox('Do You Smoke?',['Yes','No'])
yellow_fingers=st.selectbox('Do You Have Yellow Fingers?',['Yes','No'])
anxiety=st.selectbox('Do You Feel Anxiety?',['Yes','No'])
peer_pressure=st.selectbox('Do You Feel Peer Pressure?',['Yes','No'])
chronic_disease=st.selectbox('Do You Have Any Chronic Disease?',['Yes','No'])
fatigue=st.selectbox('Do You Feel Fatigue?',['Yes','No'])
allergy=st.selectbox('Do You Have Any Allergy?',['Yes','No'])
wheezing=st.selectbox('Do You Have Any Wheezing Problem?',['Yes','No'])
alcohol_consumption=st.selectbox('Do You Drink Alcohol?',['Yes','No'])
coughing=st.selectbox('Do You Have Sort Of Coughing?',['Yes','No'])
short_breath=st.selectbox("Do You Problem Of Short Breathing?",['Yes','No'])
swallow_difficulty=st.selectbox("Do You Have Swallow Difficulty?",['Yes','No'])
chest_pain=st.selectbox("Do You Have Chest Pain?",['Yes','No'])

# input_data=[gender,smoking,yellow_fingers,anxiety,peer_pressure,chronic_disease,fatigue,allergy,wheezing,alcohol_consumption,coughing,short_breath,chest_pain]
# categorical_dict={'Male':1,'Female':0,'Yes':1,'No':0}
# input_list=[categorical_dict[x] for x in input_data]
# input_list=input_list.insert(1,age/100)
# model=joblib.load('svc_model.joblib')

# def predict():
#     prediction=model.predict([[input_list]])[0]
#     st.session_state.Prediction=prediction


# if st.button(label='Predict',help="Click to predict your lung cancer risk based on the provided information."):
#   #df=pd.DataFrame(np.array([input_list]))  

#   predict()
#   if 'Prediction' not in st.session_state:
#     st.session_state.Prediction==None
#   if st.session_state.Prediction is not None:
#     if st.session_state.prediction == 1:
#         st.header(f'{name}, you have lung cancer')
#     else:
#         st.header(f'{name}, you DO NOT have lung cancer')  

# Creating a list for model
input_variables = [gender, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consumption, coughing, short_breath, swallow_difficulty, chest_pain]

model_input_dict = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
model_list = [model_input_dict[y] for y in input_variables]
model_list.insert(1, age / 100)

# Importing model and preparing dataframe for it
model = joblib.load('svc_model.joblib')
columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING','ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY', 'CHEST PAIN']
df = pd.DataFrame(np.array([model_list]), columns=columns)

# Initialize session state for prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def predict():
    prediction = model.predict(df)[0]
    st.session_state.prediction = prediction

st.button(label='Predict', on_click=predict, help="Click to predict your lung cancer risk based on the provided information.")

# Display the prediction result
if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.header(f'{name}, you have lung cancer')
    else:
        st.header(f'{name}, you DO NOT have lung cancer')
 