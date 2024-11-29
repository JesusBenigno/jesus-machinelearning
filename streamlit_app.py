import streamlit as st
import pandas as pd

st.title('WebApp para pronosticar demanda de bicicletas')

st.info('Esta aplicacion entrena un modelo de ML de renta de bicicletas y permite hacer pronosticos!')

df = pd.read_csv('https://github.com/JesusBenigno/jesus-machinelearning/blob/master/SeoulBikeData.csv')
df
