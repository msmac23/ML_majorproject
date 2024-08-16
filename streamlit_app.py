import streamlit as st
import pandas as pd

st.title('🎈 Vehicle Prediction Algorithm')

st.info('This is a Machine learning app to predict the prices of used cars on craigslist')

st.write('Good day Mr. Senior. Please see our attempt at a machine learning algorithm app.')

with st.expander('Data'):
  st.write('**Raw Vehicles Data**')
  vehicles_df = pd.read_csv('https://raw.githubusercontent.com/msmac23/ML_majorproject/master/sample_vehicles_data.csv')
  vehicles_df

  st.write('**X**')
  X = vehicles_df.drop('price', axis =1)
  X

  st.write('**y**')
  y = vehicles_df.price
  y















                  
