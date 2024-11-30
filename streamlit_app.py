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
  hour = st.slider("Hora", 0,23)
  temp = st.slider('Temp.(°C)', -25, 25, 15)
  seaso = st.selectbox('Season', ('Spring','Summer','Fall','Winter'))
  if(seaso == 'Spring'):
    season = 1
  elif(seaso == 'Summer'):
    season = 2
  elif(seaso == 'Fall'):
    season = 3
  elif(seaso == 'Winter'):
    season = 4
  weekda = st.selectbox('Weekday', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
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
  weathers = st.selectbox('Weather', ('Clear', 'Cloudy', 'Rainy', 'Thunder'))
  if(weathers == 'Clear'):
    weathersit = 1
  elif(weathers == 'Cloudy'):
    weathersit = 2
  elif(weathers == 'Rainy'):
    weathersit = 3
  elif(weathers == 'Thunder'):
    weathersit = 4
  hum = st.slider('Hum. (%)', 0.0, 1.0, 0.6)
  windspeed = st.slider('Vel. Viento (Normalizada)', 0.02, 0.5, 0.2)

# Creamos un dataframe para las features de entrada (del slider bar)
data = {'season': season,
        'weekday': weekday,
        'weathersit': weathersit,
        'temp': temp,
        'hum': hum,
        'windspeed': windspeed}
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
rf_reg = RandomForestRegressor(n_estimators= 100, random_state=0)
rf_reg.fit(X_train,y_train)
y_pred = rf_reg.predict(X_test)

#regressor = LinearRegression()
#regressor.fit(X_train,y_train)
#y_pred = regressor.predict(X_test)
r2 = r2_score(y_pred, y_test)
r2










