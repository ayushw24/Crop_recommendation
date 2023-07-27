import streamlit as st
import numpy as np
import pickle,sklearn,warnings
warnings.filterwarnings('ignore')

loaded_model=pickle.load(open('\trained_model_cyp.sav','rb'))

def cyp(input_data):
    data = np.asarray(input_data)
    prediction = loaded_model.predict(data)
    return prediction[0]

def main():
    st.title('Crop Yeild Prediction')
    
    N = st.text_input('Enter N Value')
    P = st.text_input('Enter P Value')
    K = st.text_input('Enter K Value')
    Temperature = st.text_input('Enter Temperature')
    Humidity = st.text_input('Enter Humidity')
    PH = st.text_input('Enter PH')
    Rainfall = st.text_input('Enter Rainfall')

    pred = ''
    if st.button('Predict Crop'):
        pred = cyp([[N, K, P, Temperature, Humidity, PH, Rainfall]])
    pred1=pred.upper()    
    st.success(pred1)

if __name__ == '__main__':
    main()