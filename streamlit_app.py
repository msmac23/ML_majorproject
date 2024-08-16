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

  years = (
    1910,	1928,	1929,	1930,	1931,	1935,	1937,	1940,	1947,	1948,	1949,	1952,	1953,	1954,	1955,	1957,	1958, 1959,	
    1960,	1962,	1963,	1964,	1965,	1966,	1967,	1968,	1969,	1970,	1971,	1972,	1973,	1975,	1976,	1977,	1978,	1979,
    1980,	1981,	1982,	1983,	1984,	1985,	1986,	1987,	1988,	1989,	1990,	1991,	1992,	1993,	1994,	1995,	1996,	1997,
    1998,	1999,	2000,	2001,	2002,	2003,	2004,	2005,	2006,	2007,	2008,	2009,	2010,	2011,	2012,	2013,	2014,	2015,
    2016, 2017,	2018,	2019,	2020,	2021,	2022)
    
  # Step 2: Create a slider for selecting a year
  min_year = min(years)
  max_year = max(years)

  selected_year = st.slider(
    'Select a Year',
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)  # Default value as a range
  )

  # Display the selected year
  st.write(f'Selected Year: {selected_year}') 
  
  # bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
   

  # Creating sliders and selectbox for each variable
  region = st.selectbox('Region', regions)
  













                  
