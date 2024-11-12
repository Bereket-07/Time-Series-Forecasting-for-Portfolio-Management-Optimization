from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def lstm_model(train_data, test_data, time_stamp=60):
    logger.info("LSTM model training and predictions")
    try:
        # Normalize the training data
        logger.info("Normalizing the training data")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))  # Train data scaling

        # Prepare the training dataset
        X_train, y_train = create_data_set(scaled_train_data, time_stamp)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        logger.info("Training the LSTM model")
        model.fit(X_train, y_train, batch_size=1, epochs=5)

        # Prepare the test dataset (scale it using the same scaler as the training data)
        logger.info("Normalizing the test data")
        scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))  # Test data scaling
        X_test, y_test = create_data_set(scaled_test_data, time_stamp)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Make predictions on the test data
        logger.info("Making predictions on the test data")
        predictions = model.predict(X_test)
        forecast = scaler.inverse_transform(predictions)  # Inverse transform to original scale
        print(f"The forecast result from LSTM model: {forecast}")

        return model , forecast

    except Exception as e:
        logger.error(f"Error occurred while training and predicting the LSTM model: {e}")

def create_data_set(data, time_stamp=1):
    """
    Create a dataset from the input data for supervised learning.
    This function creates sliding windows of 'time_stamp' length for the X values
    and the corresponding 'y' values (next value after the time window).
    """
    X, Y = [], []
    for i in range(len(data) - time_stamp - 1):
        X.append(data[i:(i + time_stamp), 0])
        Y.append(data[i + time_stamp, 0])
    return np.array(X), np.array(Y)
