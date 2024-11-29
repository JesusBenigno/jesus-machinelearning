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


with st.expander('**Visualizacion de Data**'):
  st.scatter_chart(data=df, x='season', y='cnt', height = 500)

# Input features por el usuario para pronosticar
with st.sidebar:
  st.header('Features de entrada')
  seaso = st.selectbox('Season', ('Spring','Summer','Fall','Winter'))
  if(seaso == 'Spring'):
    season = 1
  elif(seaso == 'Summer'):
    season = 2
  elif(seaso == 'Fall'):
    season = 3
  elif(seaso == 'Winter'):
    season = 4
  weekda = st.selectbox('Weekday', ('Monday', 'Tuesday', 'Wednesady', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
  if(weekda == 'Monday'):
    weekday = 1
  elif(weekda == 'Tuesday'):
    weekday = 2
  elif(weekda == 'Wednesday'):
    weekday = 3
  elif(weekda == 'Thursday'):
    weekday = 4
  elif(weekda == 'Friday'):
    weekday = 5
  elif(weekda == 'Saturday'):
    weekday = 6
  elif(weekda == 'Sunday'):
    weekday = 0
  weathersit = st.selectbox('Weather', ('Clear', 'Cloudy', 'Rainy', 'Heavy Rain'))
  temp = st.slider('Temp. (Normalizada)', 0.06, 0.86, 0.5)
  hum = st.slider('Hum. (%)', 0.0, 1.0, 0.6)
  windspeed = st.slider('Vel. Viento (Normalizada)', 0.02, 0.5, 0.2)

season










