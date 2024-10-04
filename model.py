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

def LSTM_model(data, company, country):

    data.replace('n/a', np.nan, inplace=True)
    data = data.fillna(method='ffill')

    features = ['Stock Price (2015)', 'Stock Price (2016)', 'Stock Price (2017)', 'Stock Price (2018)', 'Stock Price (2019)', 'Stock Price (2020)', 'Stock Price (2021)', 'Stock Price (2022)', 'Stock Price (2023)', 'Expense (2015)', 'Expense (2016)', 'Expense (2017)', 'Expense (2018)', 'Expense (2019)', 'Expense (2020)', 'Expense (2021)', 'Expense (2022)', 'Expense (2023)', 'Revenue (2015)', 'Revenue (2016)', 'Revenue (2017)', 'Revenue (2018)', 'Revenue (2019)', 'Revenue (2020)', 'Revenue (2021)', 'Revenue (2022)', 'Revenue (2023)', 'Market share (2015)', 'Market share (2016)', 'Market share (2017)', 'Market share (2018)', 'Market share (2019)', 'Market share (2020)', 'Market share (2021)', 'Market share (2022)', 'Market share (2023)']
    target_columns = ['Market share (2024)', 'Stock Price (2024)', 'Revenue (2024)', 'Expense (2024)']
    filtered_data = data[(data['Company'] == company) & (data['Country'] == country)]
    print(filtered_data[features + target_columns])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(filtered_data[features + target_columns])
    print(scaled_data)

    def create_sequences(data, time_steps = 1): 
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), :-len(target_columns)])
            y.append(data[i + time_steps, -len(target_columns):])
        return np.array(X), np.array(y)
    
    time_steps = 3
    X, y = create_sequences(scaled_data, time_steps)
    print(X.shape, y.shape)

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(len(target_columns)))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
    print(model.summary())

    last_sequence = scaled_data[-time_steps:]
    last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

    predictions = model.predict(last_sequence)
    predictions = scaler.inverse_transform(predictions)\
    
    print(predictions)

    market_share_2024 = predictions[0][0]
    stock_price_2024 = predictions[0][1]
    revenue_2024 = predictions[0][2]
    expense_2024 = predictions[0][3]

    print(f'Market share in 2024: {market_share_2024}')
    print(f'Stock price in 2024: {stock_price_2024}')
    print(f'Revenue in 2024: {revenue_2024}')
    print(f'Expense in 2024: {expense_2024}')
        
data = preprocess_data()
LSTM_model(data, 'Quinu', 'China')    
