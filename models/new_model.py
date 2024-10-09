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

def get_stock_price(data, index):
    filtered_data = data[(data['SL No'] == index)]
    stock_prices = filtered_data[['Stock Price (2015)', 'Stock Price (2016)', 'Stock Price (2017)', 'Stock Price (2018)', 'Stock Price (2019)', 'Stock Price (2020)', 'Stock Price (2021)', 'Stock Price (2022)', 'Stock Price (2023)', 'Stock Price (2024)']]
    return stock_prices

def get_market_share(data, index):
    filtered_data = data[(data["SL No"] == index)]
    market_share = filtered_data[['Market share (2015)', 'Market share (2016)', 'Market share (2017)', 'Market share (2018)', 'Market share (2019)', 'Market share (2020)', 'Market share (2021)', 'Market share (2022)', 'Market share (2023)', 'Market share (2024)']]
    return market_share

def get_revenue(data, index):
    filtered_data = data[(data["SL No"] == index)]
    revenue = filtered_data[['Revenue (2015)', 'Revenue (2016)', 'Revenue (2017)', 'Revenue (2018)', 'Revenue (2019)', 'Revenue (2020)', 'Revenue (2021)', 'Revenue (2022)', 'Revenue (2023)', 'Revenue (2024)']]
    return revenue

def get_expense(data, index):
    filtered_data = data[(data["SL No"] == index)]
    expense = filtered_data[['Expense (2015)', 'Expense (2016)', 'Expense (2017)', 'Expense (2018)', 'Expense (2019)', 'Expense (2020)', 'Expense (2021)', 'Expense (2022)', 'Expense (2023)', 'Expense (2024)']]
    return expense

def LSTM_model(index):
    data = preprocess_data()
    stock_prices = get_stock_price(data, index)
    market_share = get_market_share(data, index)
    revenue = get_revenue(data, index)
    expense = get_expense(data, index)

    stock_prices = stock_prices.T
    market_share = market_share.T
    revenue = revenue.T
    expense = expense.T
    stock_prices = stock_prices.interpolate()
    market_share = market_share.interpolate()
    revenue = revenue.interpolate()
    expense = expense.interpolate()
    # print(stock_prices)
    # print(market_share)
    # print(revenue)
    # print(expense)

    stock_prices_scaler = MinMaxScaler()
    market_share_scaler = MinMaxScaler()
    revenue_scaler = MinMaxScaler()
    expense_scaler = MinMaxScaler()

    stock_prices_scaled = stock_prices_scaler.fit_transform(stock_prices)
    market_share_scaled = market_share_scaler.fit_transform(market_share)
    revenue_scaled = revenue_scaler.fit_transform(revenue)
    expense_scaled = expense_scaler.fit_transform(expense)
    # print(stock_prices_scaled)
    # print(market_share_scaled)
    # print(revenue_scaled)
    # print(expense_scaled)

    X = np.array([stock_prices_scaled, market_share_scaled, revenue_scaled, expense_scaled])
    X = X.T
    X = X.reshape(X.shape[1], X.shape[0], X.shape[2])
    # print(X)

    X_train_stock_prices, X_test_stock_prices, y_train_stock_prices, y_test_stock_prices = train_test_split(X, stock_prices_scaled, test_size=0.1, random_state=0)
    X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(X, revenue_scaled, test_size=0.1, random_state=0)
    X_train_expense, X_test_expense, y_train_expense, y_test_expense = train_test_split(X, expense_scaled, test_size=0.1, random_state=0)
    X_train_market_share, X_test_market_share, y_train_market_share, y_test_market_share = train_test_split(X, market_share_scaled, test_size=0.1, random_state=0)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_stock_prices.shape[1], X_train_stock_prices.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=stock_prices_scaled.shape[1]))
    stock_prices_model = model

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_expense.shape[1], X_train_expense.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=expense_scaled.shape[1]))
    expense_model = model

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_revenue.shape[1], X_train_revenue.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=revenue_scaled.shape[1]))
    revenue_model = model

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_market_share.shape[1], X_train_market_share.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=market_share_scaled.shape[1]))
    market_share_model = model

    stock_prices_model.compile(optimizer='adam', loss='mean_squared_error')
    stock_prices_model.fit(X_train_stock_prices, y_train_stock_prices, epochs=100, batch_size=32)

    expense_model.compile(optimizer='adam', loss='mean_squared_error')
    expense_model.fit(X_train_expense, y_train_expense, epochs=100, batch_size=32)

    revenue_model.compile(optimizer='adam', loss='mean_squared_error')
    revenue_model.fit(X_train_revenue, y_train_revenue, epochs=100, batch_size=32)

    market_share_model.compile(optimizer='adam', loss='mean_squared_error')
    market_share_model.fit(X_train_market_share, y_train_market_share, epochs=100, batch_size=32)

    stock_prices_predictions = stock_prices_model.predict(X_test_stock_prices)
    stock_prices_predictions = stock_prices_scaler.inverse_transform(stock_prices_predictions)

    expense_predictions = expense_model.predict(X_test_expense)
    expense_predictions = expense_scaler.inverse_transform(expense_predictions)

    revenue_predictions = revenue_model.predict(X_test_revenue)
    revenue_predictions = revenue_scaler.inverse_transform(revenue_predictions)

    market_share_predictions = market_share_model.predict(X_test_market_share)
    market_share_predictions = market_share_scaler.inverse_transform(market_share_predictions)

    return {
        'stock_prices_prediction': float(stock_prices_predictions[0][0]),
        'expense_prediction': float(expense_predictions[0][0]),
        'revenue_prediction': float(revenue_predictions[0][0]),
        'market_share_prediction': float(market_share_predictions[0][0])
    }

# print(LSTM_model('Oyope', 'UAH'))
        
