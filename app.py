import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load model and transformer
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

# Prediction function
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]
    if pred > 0.5:
        return f'This booking is more likely cancelled with chances {round(pred, 2)}.'
    else:
        return f'This booking is less likely to get cancelled with chances {round(pred, 2)}.'

# Streamlit UI
def main():
    st.title('IN HOTEL GROUP - Booking Cancellation Predictor')

    # Inputs
    lt = float(st.text_input('Enter the Lead Time in Days:', value='0'))  
    mkt = st.selectbox('How the booking is made', ['Online', 'Offline'])
    mkt = 1 if mkt == 'Online' else 0

    price = float(st.text_input('Enter the price of the room:', value='0'))
    adult = st.selectbox('How many adults?', [1, 2, 3, 4])
    
    arr_m = st.slider('Month of arrival:', min_value=1, max_value=12, step=1)

    weekd_lambda = lambda x: {'Mon': 0, 'Tues': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}[x]
    arr_w = weekd_lambda(st.selectbox('Weekday of arrival?', ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))
    dep_w = weekd_lambda(st.selectbox('Weekday of departure?', ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))
    
    weekn = int(st.text_input('Enter number of weeknights in stay:', value='0'))
    wkndn = int(st.text_input('Enter number of weekend nights in stay:', value='0'))
    totan = weekn + wkndn
