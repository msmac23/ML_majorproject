# Robyn McDougall 20231887
# Machine Learning and Data Science: Final Project Summer 2024


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('🎈 Vehicle Prediction Algorithm')

st.info('This is a Machine learning app to predict the prices of used cars on craigslist')

st.write('ROBYN MCDOUGALL   20231887.')

with st.expander('Data'):
  st.write('**Raw Vehicles Data**')
  vehicles_df = pd.read_csv('https://raw.githubusercontent.com/msmac23/ML_majorproject/master/sample_vehicles_data_final.csv')
  vehicles_df

  st.write('**X**')    # target variable
  X_raw = vehicles_df.drop('rounded_price', axis =1)
  X_raw

  st.write('**y**')   # predictor variables
  y_raw = vehicles_df.rounded_price
  y_raw

# Visualization Charts

with st.expander('Data Visualization'):
  st.scatter_chart(data=vehicles_df, x='manufacturer', y='condition', color='fuel') 
  st.scatter_chart(data=vehicles_df, x='condition', y='type', color='transmission')
  st.bar_chart(data=vehicles_df, x='type', y='cylinders', x_label= 'Type', y_label='Cylinders', color='type',
               horizontal=False, stack=None, width=None, height=None, use_container_width=True)



# Input Features
with st.sidebar:
  st.header('Input Features')
  

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
  
  state = st.selectbox('State', ('ak',	'al',	'ar',	'az',	'ca',	'co',	'ct',	'dc',	'de',	'fl',	'ga',	'hi',	'ia',	'id',	'il',	
                       'in',	'ks',	'ky',	'la',	'ma',	'md',	'me',	'mi',	'mn',	'mo',	'ms',	'mt',	'nc',	'nd',	'ne',	'nh',	'nj',	
                        'nm', 'nv',	'ny',	'oh',	'ok',	'or',	'pa',	'ri',	'sc',	'sd',	'tn',	'tx',	'ut',	'va',	'vt',	'wa',	'wi',	'wv',	'wy'))

  odometer_range = st.selectbox('Odometer', ('0-20k',	'50k-100k', '100k-150k',	'150k-200k',	'200k-300k',	
                                            '20k-50k',	'300k-400k',	'400k-500k',	'500k+'))

  # Creating a DataFrame for the input features
  data = {'year': year,
          'manufacturer': manufacturer,
          'model': model,
          'condition': condition,
          'cylinders': cylinders,
          'fuel': fuel,
          'transmission': transmission,
          'drive': drive,
          'type': type,
          'state': state,
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
encode = ['manufacturer', 'model', 'condition', 'fuel', 'transmission', 'drive', 'type', 'state', 'odometer_range']
df_vehicles = pd.get_dummies(input_vehicles, columns=encode, prefix=encode)

X = df_vehicles[1:]
input_row = df_vehicles[:1]


# Encode y
target_mapper = {3000: 0, 4000:1, 5000:2, 6000:3, 7000:4, 8000:5, 9000:6, 10000:7, 11000:8, 12000:9, 13000:10, 14000:11,
                 15000:12, 16000:13, 17000:14, 18000:15, 19000:16, 20000:17, 21000:18, 22000:19, 23000:20, 24000:21, 25000:22,
                 26000:23, 27000:24, 28000:25, 29000:26, 30000:27, 31000:28, 32000:29, 33000:30, 34000:31, 35000:32, 36000:33,
                 37000:34, 38000:35, 39000:36, 40000:37, 41000:38, 42000:39, 43000:40, 44000:41, 45000:42, 46000:43, 47000:44,
                 48000:45, 49000:46, 50000:47, 51000:48, 52000:49, 53000:50}
                
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)



with st.expander('Data preparation'):
  st.write('**Encoded X (input vehicles)**')
  input_row
  st.write('**Encoded y**')
  y

# MODEL TRAINING AND INFERENCE
# training the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

# applying model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)



df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = [3000,	4000,	5000,	6000,	7000,	8000,	9000,	10000, 11000,	12000,	13000,	14000,	15000,
                               16000,	17000,	18000,	19000,	20000,	21000,	22000,	23000,	24000,	25000,	26000,	27000,
                               28000,	29000,	30000,	31000,	32000,	33000,	34000,	35000,	36000,	37000,	38000,	39000,
                               40000,	41000,	42000,	43000,	44000,	45000,	46000,	47000,	48000,	49000,	50000,	51000,	
                               52000,	53000]
df_prediction_proba.rename(columns={0:3000, 1:4000, 2:5000,	3:6000,	4:7000,	5:8000,	6:9000,	7:10000, 8:11000,	9:12000,	10:13000,	11:14000,
                                    12:15000, 13:16000,	14:17000,	15:18000,	16:19000,	17:20000,	18:21000,	19:22000,	20:23000,	21:24000,	22:25000,	
                                    23:26000,	24:27000, 25:28000,	26:29000,	27:30000,	28:31000,	29:32000,	30:33000,	31:34000,	32:35000,	33:36000,	
                                    34:37000,	35:38000,	36:39000, 37:40000,	38:41000,	39:42000,	40:43000,	41:44000,	42:45000,	43:46000,	44:47000,	
                                    45:48000,	46:49000,	47:50000,	48:51000, 49:52000,	50:53000})


# Display predicted price
st.subheader('Predicted Rounded Price')


vehicles_rounded_price = np.array([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 
                                   22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 
                                   40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000])


 
st.success(str(vehicles_rounded_price[prediction][0]))



