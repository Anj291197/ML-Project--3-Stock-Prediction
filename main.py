import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import tensorflow as tf

import yfinance as yf
from datetime import date, timedelta
today = date.today()

end_date = today.strftime("%Y-%m-%d")

d1 = date.today() - timedelta(days=360*5) #for last 5 years

start_date = d1.strftime("%Y-%m-%d")


st.title("Stock Prediction")
user_input = st.text_input("Enter Stock Ticker", "AAPL")
data = yf.download(tickers = user_input,
                  start = start_date,
                  end = end_date)

st.subheader("Data from 2018 to 2023")
st.write(data.describe())
#Visulaization

st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = data.Close.rolling(100).mean()

fig = plt.figure(figsize = (12,6))
plt.plot(ma100)

plt.plot(data.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
st.pyplot(fig)


data_training = pd.DataFrame(data["Close"][0:int(len(data)*0.7)])

data_testing = pd.DataFrame(data["Close"][int(len(data)*0.7):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0,1))

data_training_array = scaler.fit_transform(data_training)


X_train = []
Y_train = []

for i in range(100, data_training_array.shape[0]):
  X_train.append(data_training_array[i-100])
  Y_train.append(data_training_array[i,0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
model = load_model("keras_model.h5")

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)
input_data = scaler.fit_transform(final_df)

X_test = []
Y_test = []
for i in range(100, input_data.shape[0]):
  X_test.append(input_data[i-100])
  Y_test.append(input_data[i,0])

X_test, Y_test = np.array(X_test), np.array(Y_test)

y_predicted = model.predict(X_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
Y_test = Y_test * scale_factor

st.subheader("Final Closing price")
fig1 =plt.figure(figsize = (12,6))
plt.plot(Y_test, "b", label = "Original Price")
plt.plot(y_predicted, "r", label = "Predicted Price")
plt.legend()
plt.show()
st.pyplot(fig1)

