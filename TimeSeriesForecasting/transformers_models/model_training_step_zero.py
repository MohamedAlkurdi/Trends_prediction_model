import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

data_path = '../Classification/output/regions/north_america_australia/genral_labeled_data_with_relative_traffic_rates/Entertainment/extended_data/usa.csv'
data = pd.read_csv(data_path)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def preprocess_data(data, seq_length=30):
    data = data.set_index('date')
    data = data['score']
    data.index = pd.to_datetime(data.index)
    
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    train_data = data_scaled[data.index <= '2014-11-11']
    test_data = data_scaled[data.index > '2016-11-28']
    
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    return X_train, y_train, X_test, y_test, scaler

def create_model(seq_length):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

seq_length = 30

X_train, y_train, X_test, y_test, scaler = preprocess_data(data, seq_length)

model = create_model(seq_length)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

predictions = model.predict(X_test)
print("predictions:", predictions)

predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

def test_on_time_range(model, data, start_date, end_date, window_size=30):
    filtered_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
    
    if len(filtered_data) < window_size:
        raise ValueError("Not enough data in the selected time range for testing.")
    
    scaler = MinMaxScaler()
    scaler.fit(data.values.reshape(-1, 1))
    
    filtered_data_scaled = scaler.transform(filtered_data.values.reshape(-1, 1))
    
    sequences = []
    for i in range(len(filtered_data_scaled) - window_size):
        sequences.append(filtered_data_scaled[i:i+window_size])
    X_test = np.array(sequences)
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions, filtered_data[window_size:]

predictions, actual_data = test_on_time_range(
    model, 
    data.set_index('date')['score'], 
    start_date='2017-01-01', 
    end_date='2017-05-04'
)

def plot_predictions(predictions, actual_data, title="Model Predictions vs Actual Data"):
    plt.figure(figsize=(25, 5))
    plt.plot(actual_data.index, actual_data.values, label="Actual Data", color='blue')
    prediction_dates = actual_data.index[:len(predictions)]
    plt.plot(prediction_dates, predictions, label="Predictions", color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Traffic Rate")
    plt.xticks( rotation=90)
    plt.legend()
    plt.grid()
    plt.show()

plot_predictions(predictions, actual_data)