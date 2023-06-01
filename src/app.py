#pip install streamlit
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

HOME_AIRPORTS = ('LGW', 'LIS', 'LYS')
PAIRED_AIRPORTS = ('FUE', 'AMS', 'ORY')

data_name = os.getenv('trafic', 'data/traffic_10lines.parquet')
data_path = os.path.join(os.getcwd(),data_name)


print("========")
print(data_path)

df = pd.read_parquet(data_path)

st.title('Traffic Forecaster')
st.subheader('Application réalisée par AMOUSSOU Messan')
st.markdown('Cette application affiche les prévisions du trafic aérien')
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

filtered_data = (df
                 .query(f'home_airport == "{home_airport}" and paired_airport == "{paired_airport}"')
                 .groupby(['home_airport', 'paired_airport', 'date'])
                 .agg(pax_total=('pax', 'sum'))
                 .reset_index()
                 .set_index('date'))

fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['pax_total'], mode='lines', name='Passenger Count'))
fig.update_layout(title=f'Traffic between {home_airport} and {paired_airport}', xaxis_title='Date', yaxis_title='Passenger Count')

st.plotly_chart(fig)

# Fonction réalisant une prédiction du trafic 
#pip install prophet
from prophet import Prophet

def forecast_traffic(home_airport, paired_airport, forecast_date, nb_days):

    # Filtrer les données pour la route sélectionnée
    home_airport = 'LGW'
    paired_airport = 'AMS'
    filtered_data = df[(df['home_airport'] == home_airport) & (df['paired_airport'] == paired_airport)]
    train_data = filtered_data[['date', 'pax']].rename(columns={'date': 'ds', 'pax': 'y'})

    # Entraîner le modèle Prophet
    model = Prophet()
    model.fit(train_data)

    # Préparer les dates de prévision
    forecast_start_date = pd.to_datetime(forecast_date)
    forecast_end_date = forecast_start_date + pd.DateOffset(days=nb_days)
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')

    # Effectuer la prédiction
    forecast = model.predict(pd.DataFrame({'ds': forecast_dates}))

    # Afficher la prédiction
    st.write(forecast[['ds', 'yhat']])
    
    # Ajouter la prédiction dans le graphique
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Data', line=dict(dash='dash')))
    
# ... autres éléments de l'application ...
if run_forecast:
    forecast_traffic(home_airport, paired_airport, forecast_date, nb_days)
    
fig.update_layout(title='Traffic Forecast', xaxis_title='Date', yaxis_title='Passenger Count')

st.plotly_chart(fig)