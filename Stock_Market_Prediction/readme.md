# Stock Price Prediction with LSTM

# This project demonstrates how to use an LSTM (Long Short-Term Memory) network
# to predict stock prices using technical indicators like MACD (Moving Average Convergence Divergence)
# and RSI (Relative Strength Index). The code preprocesses stock data, computes technical indicators,
# and builds a deep learning model to predict stock prices.

# Requirements:
# - Python 3.x
# - TensorFlow
# - pandas
# - numpy
# - seaborn
# - matplotlib
# - scikit-learn
#
# Install the required libraries using pip:
# pip install tensorflow pandas numpy seaborn matplotlib scikit-learn

# Dataset:
# The dataset used in this project is the historical stock data for GOOGL (Google)
# retrieved from a public GitHub repository. The data includes columns such as 
# Date, Open, High, Low, Close, Adj Close, and Volume.

# Import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data Loading
df = pd.read_csv('https://raw.githubusercontent.com/brandynewanek/brandynewanek/main/GOOGL.csv')
df.head()  # Display the first few rows of the dataset

# Exploratory Data Analysis (EDA)
# Plot histograms for each numerical feature to observe its distribution
palette = ['#164B57','#3DB37F']
for feat in df.columns:
  if df[feat].dtype != 'object':
    sns.histplot(df, x=feat, color=palette[0])
    plt.show()

# Feature Engineering
# Exponentially Weighted Moving Averages (EMA)
df['ema_12'] = df['Close'].ewm(span=12).mean()
df['ema_26'] = df['Close'].ewm(span=26).mean()
df['macd_fast'] = df['ema_12'] - df['ema_26']
df['slow_macd'] = df['macd_fast'].ewm(span=9).mean()
df['macd'] = df['macd_fast'] - df['slow_macd']

# Relative Strength Index (RSI)
change = df['Close'].diff()
df['Gain'] = change.mask(change<0,0)
df['Loss'] = abs(change.mask(change>0,0))
df['AVG_Gain'] = df.Gain.rolling(14).mean()
df['AVG_Loss'] = df.Loss.rolling(14).mean()
df['RS'] = df['AVG_Gain'] / df['AVG_Loss']
df['rsi'] = 100 - (100 / (1 + df['RS']))

# Visualize Technical Indicators
# Plot RSI and MACD to observe trends
df[['rsi', 'macd']].plot(kind='line', color=palette, subplots=True, layout=(2, 1))

# Feature Engineering: Lag Features
# Create lag features for MACD and RSI to predict future stock prices based on past values
days_macd = [1,2,3,4,5,6,7,8,9,10]
days_rsi = [1,2,3,4,5,6,7,8,9,10]

for d in days_macd:
  df[f'macd_{d}_dyago'] = df['macd'].shift(d)

for d in days_rsi:
  df[f'rsi_{d}_dyago'] = df['rsi'].shift(d)

# Data Preprocessing: Standardize features
# Standardize the data for better model performance
std_scaler = StandardScaler()
x_ts = std_scaler.transform(x_ts)

# Model Setup: LSTM
# Define an LSTM model using TensorFlow/Keras
dim = x_tr.shape[1]
dense_act = 'elu'
recur_act = 'tanh'
eps = 500
lrning_rt = .001
ls = 'mae'
opt = tf.keras.optimizers.SGD(learning_rate=lrning_rt)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(13, recurrent_activation=recur_act, input_shape=[dim, 1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation=dense_act),
    tf.keras.layers.Dense(8, activation=dense_act),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(loss=ls, optimizer=opt, metrics=['mae'])

# Model Training: Fit the LSTM model
# Train the model using the training data
history = model.fit(x_tr, y_tr, validation_split=0.2, epochs=eps, batch_size=1024)

# Visualize Training History
# Plot the training and validation loss during training
pd.DataFrame(history.history).plot(color=palette)

# Notes:
# - The model may require hyperparameter tuning and further refinement to achieve better prediction performance.
# - More features or external factors (such as news sentiment) can be added to improve the model.
# - The LSTM model is computationally expensive and requires significant resources to train effectively.

