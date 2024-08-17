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
  X_raw = vehicles_df.drop('rounded_price', axis =1)
  X_raw

  st.write('**y**')   # predictor variables
  y = vehicles_df.rounded_price
  y

# Visualization Charts

with st.expander('Data Visualization'):
  st.scatter_chart(data=vehicles_df, x='manufacturer', y='condition', color='fuel') 
  st.scatter_chart(data=vehicles_df, x='condition', y='type', color='transmission')
  st.bar_chart(data=vehicles_df, x='type', y='cylinders', x_label= 'Type', y_label='Cylinders', color='type',
               horizontal=False, stack=None, width=None, height=None, use_container_width=True)



# Input Features
with st.sidebar:
  st.header('Input Features')
  
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

  
  # Creating sliders and selectbox for each variable
  region = st.selectbox('Region', regions)
  year = st.slider('Year of Car', 1910, 2022, 1981) 
  manufacturer = st.selectbox('Manufacturer', ('acura',	'audi', 'bmw',	'cadillac',	'chevrolet',	'chrysler',	'dodge',	
                                               'ford',	'gmc',	'honda',	'hyundai',	'jeep',	'kia',	'lexus',	'mercedes-benz',	
                                               'nissan',	'others',	'ram',	'subaru',	'toyota',	'volkswagen'
  ))
  
  # Filter car models based on the selected manufacturer
  filtered_models = vehicles_df[vehicles_df['manufacturer'] == manufacturer]['model'].tolist()
  model = st.selectbox('Models', filtered_models)
  
  condition = st.selectbox('Condition', ('excellent', 'fair',	'good',	'like new',	'new',	'salvage'))
  cylinders = st.slider('Cylinder', 0, 12, 6) 
  fuel = st.selectbox('Fuel Type', ('diesel',	'electric',	'gas',	'hybrid',	'other'))
  transmission = st.selectbox('Transmission Type', ('automatic',	'manual',	'other'))
  drive = st.selectbox('Drive Type', ('4wd',	'fwd',	'rwd'))
  type = st.selectbox('Vehicle Type', ('bus',	'convertible',	'coupe',	'hatchback',	'mini-van',	'offroad',	
                                       'other',	'pickup',	'sedan',	'SUV',	'truck',	'van',	'wagon'))
  odometer_range = st.selectbox('Odometer', ('0-20k',	'50k-100k', '100k-150k',	'150k-200k',	'200k-300k',	
                                            '20k-50k',	'300k-400k',	'400k-500k',	'500k+'))

  # Creating a DataFrame for the input features
  data = {'region': region,
          'year': year,
          'manufacturer': manufacturer,
          'model': model,
          'condition': condition,
          'cylinders': cylinders,
          'fuel': fuel,
          'transmission': transmission,
          'drive': drive,
          'type': type,
          'odometer_range': odometer_range}
          
  input_df = pd.DataFrame(data, index=[0])
  input_vehicles = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input vehicle data**')
  input_df
  st.write('**Merged vehicles data**')
  input_vehicles


# Data preparation
# Encode categorical columns in X
encode = ['region', 'manufacturer', 'model', 'condition', 'fuel', 'transmission', 'drive', 'type', 'odometer_range']
df_vehicles = pd.get_dummies(input_vehicles, columns=encode, prefix=encode)


#with st.expander('Data preparation'):
 # st.write('**Encoded X (input vehicles)**')
  #input_row
  #st.write('**y**')
  #y

#X = df_vehicles[1:]
#input_row = df_vehicles[:1]




#input_vehicles - used to check 


 

  # region,price,year,manufacturer,model,condition,cylinders,fuel,transmission,drive,type,odometer_bins











                  
