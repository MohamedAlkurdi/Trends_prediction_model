import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error


# Load data files
nigeria_data = pd.read_csv("../Classification/output/regions/africa/genral_labeled_data_with_relative_traffic_rates/Entertainment/Nigeria_with_relative_traffic_rates.csv")
kenya_data = pd.read_csv("../Classification/output/regions/africa/genral_labeled_data_with_relative_traffic_rates/Entertainment/Kenya_with_relative_traffic_rates.csv")
south_africa_data = pd.read_csv("../Classification/output/regions/africa/genral_labeled_data_with_relative_traffic_rates/Entertainment/SouthAfrica_with_relative_traffic_rates.csv")

# Preprocess data (e.g., normalize traffic, create time series sequences)
def preprocess_data(data):
    data = data.set_index('date')
    data = data['traffic_rate']
    data.index = pd.to_datetime(data.index) 
    X_train = data.loc[data.index <= '2016-11-28']
    y_train = data.loc[data.index > '2016-11-28']
    return X_train, y_train

# Define LSTM model
def create_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(30, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train on Nigeria's data
X_train_nigeria, y_train_nigeria = preprocess_data(nigeria_data)
print("X_train_nigeria")
print(X_train_nigeria)
print("--------------------------")
print("y_train_nigeria")
print(y_train_nigeria)
model = create_model()
model.fit(X_train_nigeria, y_train_nigeria, epochs=50, verbose=1)

# Fine-tune on Kenya's data
# X_train_kenya, y_train_kenya = preprocess_data(kenya_data)
# model.fit(X_train_kenya, y_train_kenya, epochs=30, verbose=1)

# Fine-tune on South Africa's data
X_train_south_africa, y_train_south_africa = preprocess_data(south_africa_data)
model.fit(X_train_south_africa, y_train_south_africa, epochs=30, verbose=1)

# Test on a timeline from any country (e.g., Nigeria)
X_test, y_test = preprocess_data(nigeria_data)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae}")