from django.shortcuts import render

# Create your views here.
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY = '620874e3f1d0b71fe003faa834fecb20'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

#1.Fetch Current Weather Data
def get_current_weather(city):
  url = f"{BASE_URL}weather?q={city}&appid={API_KEY }&units=metrics"
  response = requests.get(url)
  data = response.json()
  wind_gust_deg = data.get('wind', {}).get('deg')
  return {
      'city': data['name'],
      'current_temp': round(data['main']['temp']),
      'feels_like': round(data['main']['feels_like']),
      'temp_min': round(data['main']['temp_min']),
      'temp_max': round(data['main']['temp_max']),
      'humidity': round(data['main']['humidity']),
      'description': data['weather'][0]['description'],
      'country': data['sys']['country'],
      'wind_gust_deg': wind_gust_deg if wind_gust_deg is not None else 0,
      'pressure': data['main']['pressure'],
      'WindGustSpeed': data['wind']['speed'],

      'clouds': data['clouds']['all'],
      'visibility': data['visibility'],

  }

#2. Read Historical Data
def read_historical_data(filename):
  df = pd.read_csv(filename)
  df = df.dropna()
  df = df.drop_duplicates()
  return df

#3.Prepare data for training
def prepare_data(data):
  le = LabelEncoder()
  data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
  X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
  Y = data['RainTomorrow']
  return X, Y, le

#4.Train Rain Prediction Model
def train_rain_model(X, Y):
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, Y_train)
  Y_pred = model.predict(X_test)
  print("Mean Squared Error for Rain Model")
  print(mean_squared_error(Y_test, Y_pred))
  return model

#5.Prepare regression data
def prepare_regression_data(data, feature):
  X, Y = [], []
  for i in range(len(data) - 1):
    X.append(data[feature].iloc[i])
    Y.append(data[feature].iloc[i+1])
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)
    return X, Y
  
#Train Regression Model
def train_regression_model(X, Y):
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X, Y)
  return model

#Predict Future
def predict_future(model, current_value):
  prediction = [current_value]
  for i in range(5):
    next_value = model.predict(np.array([[prediction[-1]]]))
    return prediction[1:]
  
#Weather Analysis Prediction
def weather_view(request):
      if request.method =='POST':
       city = request.POST.get('city')
       current_weather = get_current_weather(city)
       wind_gust_deg = current_weather['wind_gust_deg'] % 360
       csv_path = os.path.join('C:\\Users\\hongo\\MachineLearningProject\\weather.csv')
       historical_data = read_historical_data(csv_path)
       X, Y, le = prepare_data(historical_data)
       rain_model = train_rain_model(X, Y)
       
       compass_points = [
       ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
       ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
       ("SE", 123.75, 146.75), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
       ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
       ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
       ("NNW", 326.25, 348.75)

  ]
       compass_direction = next(point for point, start, end in compass_points if start <= wind_gust_deg < end)
       compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1
       current_data = {
         'MinTemp': current_weather['temp_min'],
         'MaxTemp': current_weather['temp_max'],
         'WindGustDir': compass_direction_encoded,
         'WindGustSpeed': current_weather['WindGustSpeed'],
         'Humidity': current_weather['humidity'],
         'Pressure': current_weather['pressure'],
         'Temp': current_weather['current_temp'],

  }
       current_df = pd.DataFrame([current_data])
       rain_prediction = rain_model.predict(current_df)[0]
       X_temp, Y_temp = prepare_regression_data(historical_data, 'Temp')
       X_hum, Y_hum = prepare_regression_data(historical_data, 'Humidity')
       temp_model = train_regression_model(X_temp, Y_temp)
       hum_model = train_regression_model(X_hum, Y_hum )
       future_temp = predict_future(temp_model, current_weather['temp_min'])
       future_humidity = predict_future(hum_model, current_weather['humidity'])
       timezone = pytz.timezone('Asia/Karachi')
       now = datetime.now(timezone)
       next_hour = now + timedelta(hours=1)
       next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

       future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

       #store each value separaetely

       time1, time2, time3, time4, time5 = future_times
       temp1, temp2, temp3, temp4, temp5 = future_temp 
       hum1, hum2, hum3, hum4, hum5 = future_humidity

       #pass data to template

       context = {
         'location': city,
         'current_temp': current_weather['current_temp'],
         'MinTemp': current_weather['temp_min'],
         'MaxTemp': current_weather['temp_max'],
         'feels_like': current_weather['feels_like'],
         'humidity': current_weather['humidity'],
         'clouds': current_weather['clouds'],
         'description': current_weather['description'],
         'city': current_weather['city'],
         'country': current_weather['country'],
         'time': datetime.now(),
         'date': datetime.now().strftime("%B %d, %Y"),
         'wind': current_weather['WindGustSpeed'],
         'pressure': current_weather['pressure'],
         'visibility': current_weather['visibility'],
         'rain_prediction': "Yes" if rain_prediction else "No",

         'time1': time1,
         'time2': time2,
         'time3': time3,
         'time4': time4,
         'time5': time5,

         'temp1': f"{round(temp1, 1)}",
         'temp2': f"{round(temp2, 1)}",
         'temp3': f"{round(temp3, 1)}",
         'temp4': f"{round(temp4, 1)}",
         'temp5': f"{round(temp5, 1)}",

         'hum1': f"{round(hum1, 1)}",
         'hum2': f"{round(hum2, 1)}",
         'hum3': f"{round(hum3, 1)}",
         'hum4': f"{round(hum4, 1)}",
         'hum5': f"{round(hum5, 1)}",
       }

       return render(request, 'weather.html', context)
      
      return render(request, 'weather.html') 

  