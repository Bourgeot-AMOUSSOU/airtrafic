import pandas as pd
import streamlit as st

HOME_AIRPORTS = ('LGW', 'LIS', 'LYS')
PAIRED_AIRPORTS = ('FUE', 'AMS', 'ORY')

df = pd.read_parquet('data/traffic_10lines.parquet')

st.title('Traffic Forecaster')

with st.sidebar:
    home_airport = st.selectbox(
        'Home Airport', HOME_AIRPORTS)
    paired_airport = st.selectbox(
        'Paired Airport', PAIRED_AIRPORTS)
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 7, 30, 1)
    run_forecast = st.button('Forecast')
    
st.write('Home Airport selected:', home_airport)
st.write('Paired Airport selected:', paired_airport)
st.write('Days of forecast:', nb_days)
st.write('Date selected:', forecast_date)


# Affichage de la table
st.dataframe(data=df, width=600, height=300)