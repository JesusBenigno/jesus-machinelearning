import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.title('WebApp para pronosticar demanda de bicicletas')

st.info('Esta aplicacion entrena un modelo de ML de renta de bicicletas y permite hacer pronosticos!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/JesusBenigno/jesus-machinelearning/refs/heads/master/data/new_data.csv')
  df
  
# Definir X_raw (season, weekday, weathersit, temp, hum, windspeed)
x_raw = df.drop(['Rented Bike Count'], axis=1)
# Definir Y_raw (cnt)
y_raw = df['Rented Bike Count'].values

with st.expander('Visualizacion de Data'):
  st.scatter_chart(data=df, x='Temperature(°C)', y='Rented Bike Count', height = 500, color="#1C7643")

# Input features por el usuario para pronosticar
with st.sidebar:
  st.header('Features de entrada')
  hour = st.slider("Hora", 0,23,12)
  temp = st.slider('Temp.(°C)', -25, 25, 15)
  hum = st.slider('Hum. (%)', 0.0, 1.0, 0.5)
  windspeed = st.slider('Vel. Viento(m/s)', 0.0, 7.4, 3.5)
  visibility = st.slider('Visibilidad',27,2000,1000)
  dew_point = st.slider('Temp. Punto de Rocio(°C)',-36.6,27.2,0.0)
  solar_rad = st.slider('Radiacion solar(MJ/m2)', 0.0, 3.52, 1.5)
  rain = st.slider('Lluvia(mm)',0,35,15)
  snow = st.slider('Nieve(cm)',0,10,5)
  seaso = st.selectbox('Season', ('Spring','Summer','Fall','Winter'))
  if(seaso == 'Spring'):
    season = 1
  elif(seaso == 'Summer'):
    season = 2
  elif(seaso == 'Winter'):
    season = 3
  elif(seaso == 'Fall'):
    season = 0
  holid = st.selectbox('Dia Feriado?',('Si','No'))
  if(holid == 'Si'):
    holiday = 0
  elif(holid == 'No'):
    holiday = 1
  diafu = st.selectbox('Servicio de bici activo?',('Si','No'))
  if(diafu == 'Si'):
    diafun = 1
  elif(diafu == 'No'):
    diafun= 0
    
# Creamos un dataframe para las features de entrada (del slider bar)
data = {'Hour': hour,
        'Temperature(°C)': temp,
        'Humidity(%)': hum,
        'Wind speed (m/s)': windspeed,
        'Visibility (10m)': visibility,
        'Dew point temperature(°C)': dew_point,
        'Solar Radiation (MJ/m2)': solar_rad,
        'Rainfall(mm)': rain,
       'Snowfall (cm)': snow,
       'Seasons': season,
       'Holiday': holiday,
       'Functioning Day': diafun}
input_df = pd.DataFrame(data, index=[0])
input_bike = pd.concat([input_df, x_raw], axis=0)

with st.expander('Features de entrada'):
  st.write('**Nuevos parametros ingresados**')
  input_df
  st.write('**Datos conbinados**')
  input_bike

# Entrenar modelo y predecir con las features de entrada
X_train, X_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.20, random_state=1, shuffle=True)

st.write('**Regresion Lineal Multiple**')
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
st.write('R2 score: ')
r2 = r2_score(y_test, y_pred)
r2
input = 1
inputs = [[input, hour, temp, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]]
inputs








