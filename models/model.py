import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def preprocess_data():
    data = pd.read_csv('data.csv')

    def convert_to_float(value):
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '')
            if 'B' in value:
                return float(value.replace('B', '')) * 1e9
            elif 'M' in value:
                return float(value.replace('M', '')) * 1e6
            elif 'K' in value:
                return float(value.replace('K', '')) * 1e3
            else:
                return float(value)
        return value

    financial_columns = [
        'Market Cap',
        'Stock Price (2015)', 'Stock Price (2016)', 'Stock Price (2017)', 'Stock Price (2018)',
        'Stock Price (2019)', 'Stock Price (2020)', 'Stock Price (2021)', 'Stock Price (2022)',
        'Stock Price (2023)', 'Stock Price (2024)',
        'Expense (2015)', 'Expense (2016)', 'Expense (2017)', 'Expense (2018)', 'Expense (2019)',
        'Expense (2020)', 'Expense (2021)', 'Expense (2022)', 'Expense (2023)', 'Expense (2024)',
        'Revenue (2015)', 'Revenue (2016)', 'Revenue (2017)', 'Revenue (2018)', 'Revenue (2019)',
        'Revenue (2020)', 'Revenue (2021)', 'Revenue (2022)', 'Revenue (2023)', 'Revenue (2024)',
        'Market share (2015)', 'Market share (2016)', 'Market share (2017)', 'Market share (2018)',
        'Market share (2019)', 'Market share (2020)', 'Market share (2021)', 'Market share (2022)',
        'Market share (2023)', 'Market share (2024)'
    ]
    
    for col in financial_columns:
        data[col] = data[col].apply(convert_to_float)

    return data

def get_historical_data(data, index, column_prefix):
    filtered_data = data[data['SL No'] == index]
    columns = [f'{column_prefix} ({year})' for year in range(2015, 2025)]
    historical_data = filtered_data[columns]
    return historical_data

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicting next year’s single value
    return model

def scale_and_reshape(data, scaler):
    data_scaled = scaler.fit_transform(data)
    data_scaled = np.expand_dims(data_scaled, axis=0)  # LSTM expects 3D input
    return data_scaled

def LSTM_model(index):
    data = preprocess_data()

    stock_prices = get_historical_data(data, index, 'Stock Price')
    market_share = get_historical_data(data, index, 'Market share')
    revenue = get_historical_data(data, index, 'Revenue')
    expense = get_historical_data(data, index, 'Expense')

    # Interpolating any missing data
    stock_prices = stock_prices.T.interpolate().T
    market_share = market_share.T.interpolate().T
    revenue = revenue.T.interpolate().T
    expense = expense.T.interpolate().T

    # Scaling data
    stock_prices_scaler = MinMaxScaler()
    market_share_scaler = MinMaxScaler()
    revenue_scaler = MinMaxScaler()
    expense_scaler = MinMaxScaler()

    stock_prices_scaled = scale_and_reshape(stock_prices.values.T, stock_prices_scaler)
    market_share_scaled = scale_and_reshape(market_share.values.T, market_share_scaler)
    revenue_scaled = scale_and_reshape(revenue.values.T, revenue_scaler)
    expense_scaled = scale_and_reshape(expense.values.T, expense_scaler)

    # Input shape for LSTM (samples, timesteps, features)
    input_shape = (stock_prices_scaled.shape[1], stock_prices_scaled.shape[2])

    # Building models for each metric
    stock_prices_model = build_lstm_model(input_shape)
    expense_model = build_lstm_model(input_shape)
    revenue_model = build_lstm_model(input_shape)
    market_share_model = build_lstm_model(input_shape)

    # Compiling models
    for model in [stock_prices_model, expense_model, revenue_model, market_share_model]:
        model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the models with historical data
    stock_prices_model.fit(stock_prices_scaled, stock_prices_scaled[:, -1, :], epochs=100, batch_size=32)
    expense_model.fit(expense_scaled, expense_scaled[:, -1, :], epochs=100, batch_size=32)
    revenue_model.fit(revenue_scaled, revenue_scaled[:, -1, :], epochs=100, batch_size=32)
    market_share_model.fit(market_share_scaled, market_share_scaled[:, -1, :], epochs=100, batch_size=32)

    # Predict next year's data (2025)
    stock_prices_next_year = stock_prices_model.predict(stock_prices_scaled)
    expense_next_year = expense_model.predict(expense_scaled)
    revenue_next_year = revenue_model.predict(revenue_scaled)
    market_share_next_year = market_share_model.predict(market_share_scaled)

    # Inverse transform the predictions to get the actual scale
    stock_prices_next_year = stock_prices_scaler.inverse_transform(stock_prices_next_year)[0][0]
    expense_next_year = expense_scaler.inverse_transform(expense_next_year)[0][0]
    revenue_next_year = revenue_scaler.inverse_transform(revenue_next_year)[0][0]
    market_share_next_year = market_share_scaler.inverse_transform(market_share_next_year)[0][0]

    return {
        'stock_prices_prediction_2025': float(stock_prices_next_year),
        'expense_prediction_2025': float(expense_next_year),
        'revenue_prediction_2025': float(revenue_next_year),
        'market_share_prediction_2025': float(market_share_next_year)
    }

# Example usage
# print(LSTM_model(1))
