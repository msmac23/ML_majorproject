import streamlit as st
import pandas as pd

st.title('🎈 Vehicle Prediction Algorithm')

st.info('This is a Machine learning app to predict the prices of used cars on craigslist')

st.write('Good day Mr. Senior. Please see our attempt at a machine learning algorithm app.')

with st.expander('Data'):
  st.write('**Raw Vehicles Data**')
  vehicles_df = pd.read_csv('https://raw.githubusercontent.com/msmac23/ML_majorproject/master/sample_vehicles_data.csv')
  vehicles_df

  st.write('**X**')    # target variable
  X = vehicles_df.drop('price', axis =1)
  X

  st.write('**y**')   # predictor variables
  y = vehicles_df.price
  y

# Visualization Charts

with st.expander('Data Visualization'):
  st.scatter_chart(data=vehicles_df, x='manufacturer', y='condition', color='fuel') 
  st.scatter_chart(data=vehicles_df, x='condition', y='type', color='transmission')
  st.bar_chart(data=vehicles_df, x='type', y='cylinders', x_label= 'Type', y_label='Cylinders', color='type',
               horizontal=False, stack=None, width=None, height=None, use_container_width=True)


# User Input
with st.sidebar:
  st.header('Input Features')
  # region,price,year,manufacturer,model,condition,cylinders,fuel,odometer,title_status,
  #transmission,type,paint_color,drive_4wd,drive_fwd,drive_rwd,geo_location
       # Define the regions for the selectbox
  regions = (
    'atlanta', 'austin', 'baltimore', 'boston', 'central NJ', 'charlotte',
    'chicago', 'cincinnati', 'colorado springs', 'columbus', 'dallas / fort worth',
    'denver', 'des moines', 'detroit metro', 'ft myers / SW florida', 'grand rapids',
    'houston', 'inland empire', 'jacksonville', 'kansas city', 'las vegas',
    'long island', 'los angeles', 'milwaukee', 'minneapolis / st paul', 'nashville',
    'new york city', 'north jersey', 'oklahoma city', 'orange county', 'orlando',
    'others', 'phoenix', 'pittsburgh', 'portland', 'raleigh / durham / CH',
    'reno / tahoe', 'rochester', 'sacramento', 'san diego', 'seattle-tacoma',
    'SF bay area', 'south florida', 'south jersey', 'spokane / coeur d\'alene',
    'st louis, MO', 'stockton', 'tampa bay area', 'tucson', 'washington, DC'
  )

  # Create a selectbox for region selection
  region = st.selectbox('Region', regions)












                  
