import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

st.title('🚲 WebApp para pronosticar demanda de Bicis 🚲')

st.info('Esta aplicacion entrena un modelo de ML de renta de bicicletas y permite hacer pronosticos! 💻')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('./data/new_data.csv')
  df
  
# Definir X_raw (season, weekday, weathersit, temp, hum, windspeed)
x_raw = df.drop(['Unnamed: 0','Rented Bike Count'], axis=1)
# Definir Y_raw (cnt)
y_raw = df['Rented Bike Count'].values

with st.expander('Visualizacion de Data'):
  st.scatter_chart(data=df, x='Temperature(°C)', y='Rented Bike Count', height = 500, color="#1C7643")

# Input features por el usuario para pronosticar
with st.sidebar:
  st.header('Features de entrada')
  hour = st.slider("Hora", 0,23,8)
  temp = st.slider('Temp.(°C)', -25.0, 40.0, 18.2)
  hum = st.slider('Humedad (%)', 0.0, 100.0, 63.0)
  windspeed = st.slider('Vel. Viento(m/s)', 0.0, 7.4, 0.8)
  visibility = st.slider('Visibilidad',27,2000,1731)
  dew_point = st.slider('Temp. Punto de Rocio(°C)',-36.6,27.2,11.0)
  solar_rad = st.slider('Radiacion solar(MJ/m2)', 0.00, 3.52, 1.00)
  rain = st.slider('Lluvia(mm)',0.0,35.0,0.0)
  snow = st.slider('Nieve(cm)',0.0,10.0,0.0)
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

st.subheader('Regresion Lineal Multiple')
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
st.write('R2 score: ')
r2 = r2_score(y_test, y_pred)
r2
inputs = [[hour, temp, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]]
inputs = np.array(inputs)
st.write('Prediccion de bicicletas en uso [Regression Lineal Multiple]:')
result = regressor.predict([[hour, temp, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]])
updated_res = result.flatten().astype(float)
st.success(int(updated_res))
#Mostrar grafica
arr_predicts = []
#temps = np.array([-20,-15,-10,-5,0,5,10,15,20,25,30,35,40])
temps = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6])
for i in temps:
  res_for = regressor.predict([[hour, i, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]])
  arr_predicts.append(res_for)

fig,ax = plt.subplots()
ax.scatter(x_raw['Temperature(°C)'], y_raw, color = 'red')
ax.plot(temps, arr_predicts, color = 'blue')
plt.title('Verdad o mentira (Regresion Lineal Multiple Temp. vs Bicis Rentadas)')
plt.xlabel('Temp.(°C)')
plt.ylabel('Bicis Rentadas')
st.pyplot(fig)

st.subheader('Regresion Lineal Polinomial')
grado = st.selectbox('Grado del Polinomio: (optimo=2)', (1,2,3))
poly_reg = PolynomialFeatures(degree=grado)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
# Prediccion de todos los datos para onbtener R2
X_test_PR = poly_reg.fit_transform(X_test)
y_predPR = lin_reg_2.predict(X_test_PR)
st.write('R2 score: ')
r2_PR = r2_score(y_test,y_predPR)
r2_PR
st.write('Prediccion de bicicletas en uso [Regression Lineal Polinomial]:')
resPR = lin_reg_2.predict(poly_reg.fit_transform([[hour, temp, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]]))
st.success(int(resPR))

#Mostrar grafica
arr_predicts1 = []
for j in temps:
  res_for = lin_reg_2.predict(poly_reg.fit_transform([[hour, temp, j, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]]))
  arr_predicts1.append(res_for)
  
fig1,ax1 = plt.subplots()
ax1.scatter(x_raw['Humidity(%)'], y_raw, color = 'red')
ax1.plot(temps, arr_predicts1, color = 'blue')
plt.title('Verdad o mentira (Regresion Lineal Polinomial Temp. vs Bicis Rentadas)')
plt.xlabel('Hum')
plt.ylabel('Bicis Rentadas')
st.pyplot(fig1)

st.subheader('Support Vector Regression')
kernel_svr = st.selectbox('Selecciona el kernel: (optimo=rbf)', ('rbf','linear','sigmoid'))
c_svr = st.slider('Selecciona el parametro de regularizacion C (optimo=9)', 1, 10, 9)
sc = StandardScaler()
X_train_escaled = sc.fit_transform(X_train)
X_test_escaled = sc.fit_transform(X_test)
svr_reg = SVR(kernel=kernel_svr, C=c_svr)
svr_reg.fit(X_train_escaled,y_train)
y_predSVR = svr_reg.predict(X_test_escaled)
st.write('R2 score: ')
r2_scoreSVR = r2_score(y_test,y_predSVR)
r2_scoreSVR

st.write('Prediccion de bicicletas en uso [Support Vector Regression]:')
resSVR = svr_reg.predict(sc.transform([[hour, temp, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]]))
st.success(int(resSVR))

#Mostrar grafica
arr_predicts2 = []
for j in temps:
  res_for = svr_reg.predict(sc.transform([[hour, j, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]]))
  arr_predicts2.append(res_for)
fig2,ax2 = plt.subplots()
ax2.scatter(x_raw['Temperature(°C)'], y_raw, color = 'red')
ax2.plot(temps, arr_predicts2, color = 'blue')
plt.title('Verdad o mentira (Regresion SVR Temp. vs Bicis Rentadas)')
plt.xlabel('Temp.(°C)')
plt.ylabel('Bicis Rentadas')
st.pyplot(fig2)

st.subheader('Random Forest Regression')
n_estim = st.slider('Selecciona el numero de estimadores (optimo=100)', 1, 150, 100)
rand_state = st.slider('Selecciona el random state (optimo=0)', 0, 42, 0)
RFReg = RandomForestRegressor(n_estimators = n_estim, random_state=rand_state)
RFReg.fit(X_train, y_train)
y_predRFR = RFReg.predict(X_test)
r2_RFR = r2_score(y_predRFR, y_test)
st.write('R2 score: ')
r2_RFR

st.write('Prediccion de bicicletas en uso [Random Forest Regression]:')
resRFR = RFReg.predict([[hour, temp, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]])
st.success(int(resRFR))

#Mostrar grafica
arr_predicts3 = []
for j in temps:
  res_for = RFReg.predict([[hour, j, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]])
  arr_predicts3.append(res_for)
fig3,ax3 = plt.subplots()
ax3.scatter(x_raw['Temperature(°C)'], y_raw, color = 'red')
ax3.plot(temps, arr_predicts3, color = 'blue')
plt.title('Verdad o mentira (Random Forest Temp. vs Bicis Rentadas)')
plt.xlabel('Temp.(°C)')
plt.ylabel('Bicis Rentadas')
st.pyplot(fig3)

st.subheader('Decision Tree Regression')
crit = st.selectbox('Criterio del split: (optimo=absolute error)', ('squared_error','friedman_mse','absolute_error','poisson'))
rand_state_dtr = st.slider('Selecciona el random state (optimo=42)', 0, 42, 42)
dt_reg = DecisionTreeRegressor(criterion=crit, random_state=rand_state_dtr)
dt_reg.fit(X_train,y_train)
y_predDTR = dt_reg.predict(X_test)
r2DTR = r2_score(y_predDTR, y_test)
st.write('R2 score: ')
r2DTR

st.write('Prediccion de bicicletas en uso [Decision Tree Regression]:')
resDTR = dt_reg.predict([[hour, temp, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]])
st.success(int(resDTR))

#Mostrar grafica
arr_predicts4 = []
for j in temps:
  res_for = dt_reg.predict([[hour, j, hum, windspeed, visibility, dew_point, solar_rad, rain, snow, season, holiday, diafun]])
  arr_predicts4.append(res_for)
fig4,ax4 = plt.subplots()
ax4.scatter(x_raw['Temperature(°C)'], y_raw, color = 'red')
ax4.plot(temps, arr_predicts4, color = 'blue')
plt.title('Verdad o mentira (Decision Tree Temp. vs Bicis Rentadas)')
plt.xlabel('Temp.(°C)')
plt.ylabel('Bicis Rentadas')
st.pyplot(fig4)
