#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd

def get_dataframe():
    df = pd.read_csv('data.csv')
    df = df.rename(columns={'Country Code': 'Country_Code'})

    def convert_to_num(value):
        if isinstance(value, str):
            if 'B' in value:
                return float(value.replace('$', '').replace('B', '').strip()) * 1e9  # Billion
            elif 'M' in value:
                return float(value.replace('$', '').replace('M', '').strip()) * 1e6  # Million
        return value 

    cc = df.filter(like='Stock Price').columns
    df[cc] = df[cc].map(convert_to_num)
    cc = df.filter(like='Market Share').columns
    df[cc] = df[cc].map(convert_to_num)
    cc = df.filter(like='Revenue').columns
    df[cc] = df[cc].map(convert_to_num)
    cc = df.filter(like='Expense').columns
    df[cc] = df[cc].map(convert_to_num)
    
    print((df.loc[5,'Stock Price (2024)']))
    return df
df = get_dataframe()


# In[91]:


df.head()


# In[119]:


def findthings(Comp, Country):
    key_row_df = df[(df['Company'] == Comp) & (df['Country_Code'] == Country)]
    
    if key_row_df.empty:
        return 
        
    key_row = key_row_df.iloc[0]
    # print(key_row)
    key_country_code = key_row['Country_Code']
    
    # a - How many companies are there in the same country
    freq = df[df['Country_Code'] == key_country_code].shape[0]
    
    # b - How many companies with greater diversity are in the same country
    moreDiv = df[(df['Country_Code'] == key_country_code) & (df['Diversity'] > key_row['Diversity'])].shape[0]
    
    # c - Inc/Dec in <params> (here we send the actual values, and can find diff array when graphing)
    stock_prices = key_row_df.filter(like='Stock Price').iloc[0]
    market_shares = key_row_df.filter(like='Market share').iloc[0]
    revenues = key_row_df.filter(like='Revenue').iloc[0]
    expenses = key_row_df.filter(like='Expense').iloc[0]

    
    # d - how many companies have greater <params> than this company
    ans = []
    params = ['Stock Price', 'Market share', 'Revenue', 'Expense']
    for param in params:
        latest_param_val = key_row.filter(like=param).iloc[-1]
        
        # domestically
        n_greater_val_dom = df[(df['Country_Code'] == key_country_code) & 
        (df.filter(like=param).iloc[:, -1] > latest_param_val)].shape[0]
        
        # globally
        n_greater_val_glob = df[(df.filter(like=param).iloc[:, -1] > latest_param_val)].shape[0]

        ans.append({'param':param, 'dom_cnt': n_greater_val_dom, 'glob_cnt': n_greater_val_glob})
    # print(ans)


    # lat = key_row.filter(like="Market share").iloc[-1]
    # n_greater_market_share = df[(df['Country_Code'] == key_country_code) & 
    # (df.filter(like='Market share').iloc[:, -1] > lat)]
    # print(lat,"\n", n_greater_market_share[['Market share (2024)']])
    
    return {"a": freq, "b": moreDiv, "c":[stock_prices, market_shares, revenues, expenses],"d":ans}
# findthings('Oyope','UAH')

