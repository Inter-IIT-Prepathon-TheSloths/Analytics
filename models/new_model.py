import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to convert string values to floats, considering units like K, M, and B
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

# Function to preprocess the data for all companies
def preprocess_data_for_all_companies():
    data = pd.read_csv('data.csv')
    
    financial_columns = [
        'SL No',  # Add 'SL No' column to group data by company
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
    
    # Apply the conversion function to each column
    for col in financial_columns[1:]:
        data[col] = data[col].apply(convert_to_float)

    return data

# Function to get historical data for a specific company
def get_historical_data(data, index, column_prefix):
    filtered_data = data[data['SL No'] == index]
    columns = [f'{column_prefix} ({year})' for year in range(2015, 2025)]
    historical_data = filtered_data[columns]
    return historical_data

# Function to get data for all companies
def get_all_companies_data(data, column_prefix):
    companies_data = []
    for company_index in data['SL No'].unique():
        company_data = get_historical_data(data, company_index, column_prefix)
        companies_data.append(company_data)
    return companies_data

# Function to prepare data for training (scaling and reshaping)
def prepare_data_for_training(data, column_prefix):
    companies_data = get_all_companies_data(data, column_prefix)
    
    # Stack all companies' data together
    combined_data = np.concatenate([company_data.values for company_data in companies_data], axis=0)
    
    # Scaling the data
    scaler = MinMaxScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    
    # Reshaping data for LSTM (samples, timesteps, features)
    combined_data_scaled = np.expand_dims(combined_data_scaled, axis=2)  # Adding feature dimension
    return combined_data_scaled, scaler

# Function to build an LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicting next year's single value
    return model

# Function to train the LSTM model for all companies' data
def train_lstm_model_for_all_companies():
    data = preprocess_data_for_all_companies()

    # Prepare data for training
    stock_prices, stock_prices_scaler = prepare_data_for_training(data, 'Stock Price')
    expense, expense_scaler = prepare_data_for_training(data, 'Expense')
    revenue, revenue_scaler = prepare_data_for_training(data, 'Revenue')
    market_share, market_share_scaler = prepare_data_for_training(data, 'Market share')

    # Interpolating any missing data
    # Interpolating missing data in stock_prices, market_share, revenue, and expense
    stock_prices = stock_prices.squeeze()  # Remove the extra dimension (1, 10, 1000) -> (10, 1000)
    stock_prices = pd.DataFrame(stock_prices)  # Convert to DataFrame for interpolation
    stock_prices = stock_prices.interpolate(axis=1).values  # Interpolate across rows (years)
    stock_prices = stock_prices.T  # Transpose back to the original shape
    stock_prices = np.expand_dims(stock_prices, axis=0)

    market_share = market_share.squeeze()
    market_share = pd.DataFrame(market_share)
    market_share = market_share.interpolate(axis=1).values
    market_share = market_share.T
    market_share = np.expand_dims(market_share, axis=0)

    revenue = revenue.squeeze()
    revenue = pd.DataFrame(revenue)
    revenue = revenue.interpolate(axis=1).values
    revenue = revenue.T
    revenue = np.expand_dims(revenue, axis=0)

    expense = expense.squeeze()
    expense = pd.DataFrame(expense)
    expense = expense.interpolate(axis=1).values
    expense = expense.T
    expense = np.expand_dims(expense, axis=0)

    # Input shape for LSTM (samples, timesteps, features)
    input_shape = (stock_prices.shape[1], stock_prices.shape[2])

    # Building the model
    model = build_lstm_model(input_shape)

    # Compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on all companies' data
    model.fit(stock_prices, stock_prices[:, -1, :], epochs=100, batch_size=32)
    model.fit(expense, expense[:, -1, :], epochs=100, batch_size=32)
    model.fit(revenue, revenue[:, -1, :], epochs=100, batch_size=32)
    model.fit(market_share, market_share[:, -1, :], epochs=100, batch_size=32)

    # Save the model for future use (optional)
    model.save("company_predictions_model.h5")
    return model

# Function to predict next year's data (2025) for a specific company
def predict_next_year_for_company(index):
    data = preprocess_data_for_all_companies()

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

    stock_prices_scaled = stock_prices_scaler.fit_transform(stock_prices.values.T)
    market_share_scaled = market_share_scaler.fit_transform(market_share.values.T)
    revenue_scaled = revenue_scaler.fit_transform(revenue.values.T)
    expense_scaled = expense_scaler.fit_transform(expense.values.T)

    # Reshaping data for LSTM input
    stock_prices_scaled = np.expand_dims(stock_prices_scaled, axis=0)  # LSTM expects 3D input
    market_share_scaled = np.expand_dims(market_share_scaled, axis=0)
    revenue_scaled = np.expand_dims(revenue_scaled, axis=0)
    expense_scaled = np.expand_dims(expense_scaled, axis=0)

    # Input shape for LSTM
    input_shape = (stock_prices_scaled.shape[1], stock_prices_scaled.shape[2])

    # Load the pre-trained model
    model = load_model("company_predictions_model.h5")

    # Predict the next year's data (2025)
    stock_prices_next_year = model.predict(stock_prices_scaled)
    expense_next_year = model.predict(expense_scaled)
    revenue_next_year = model.predict(revenue_scaled)
    market_share_next_year = model.predict(market_share_scaled)

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
# Train the model for all companies
# model = train_lstm_model_for_all_companies()

# Predict for a specific company (e.g., company with SL No 1)
# predictions = predict_next_year_for_company(1)
# print(predictions)
