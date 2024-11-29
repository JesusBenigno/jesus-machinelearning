import streamlit as st
import pandas as pd
import numpy as np

st.title('WebApp para pronosticar demanda de bicicletas')

st.info('Esta aplicacion entrena un modelo de ML de renta de bicicletas y permite hacer pronosticos!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/JesusBenigno/jesus-machinelearning/refs/heads/master/data/day.csv')
  df
  
# Definir X_raw (season, weekday, weathersit, temp, hum, windspeed)
x_raw = df.drop('cnt', axis=1)

# Definir Y_raw (cnt)
y_raw = df.cnt

with st.expander('Visualizacion de Data'):
  st.scatter_chart(data=df, x='weekday', y='cnt', color = 'season', height = 500)


