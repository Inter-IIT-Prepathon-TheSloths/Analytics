#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd


SUBJECT_COLS = ['Stock Price','Market Share','Revenue','Expense']

def get_dataframe():
    df = pd.read_csv('data.csv')
    df.columns= df.columns.str.replace('Market share','Market Share')

    def convert_to_num(value):
        if isinstance(value, str):
            if 'B' in value:
                return float(value.replace('$', '').replace('B', '').strip()) * 1e9  # Billion
            elif 'M' in value:
                return float(value.replace('$', '').replace('M', '').strip()) * 1e6  # Million
            elif 'K' in value:
                return float(value.replace('$', '').replace('K', '').strip()) * 1e3
            else:
                return float(value.replace('$', '').strip())
        return value 

    for col in SUBJECT_COLS:
        cc = df.filter(like=col).columns
        df[cc] = df[cc].map(convert_to_num)
    
    return df


df = get_dataframe()


# In[62]:


def get_expense_to_revenue_ratio(expenses: pd.DataFrame, revenue: pd.DataFrame):
    expense_to_revenue_ratio = {}
    for exp_col, rev_col in zip(expenses.index, revenue.index):
        expense_to_revenue_ratio[rev_col.replace('Revenue', 'Expense_to_Revenue')] = revenue[rev_col] / expenses[exp_col]

    expense_to_revenue_series = pd.Series(expense_to_revenue_ratio)

    mean_value = expense_to_revenue_series.mean()
    std_dev_value = expense_to_revenue_series.std()
    return mean_value, std_dev_value

def calculate_cagr(start_value, end_value, years):
    return (end_value / start_value) ** (1 / years) - 1


# In[63]:


def get_analytics(Comp, Country):
    key_row_df = df[(df['Company'] == Comp) & (df['Country Code'] == Country)]
    
    if key_row_df.empty:
        return {}
        
    key_row = key_row_df.iloc[0]
    # print(key_row)
    key_country_code = key_row['Country Code']
    
    # a - How many companies are there in the same country
    freq = df[df['Country Code'] == key_country_code].shape[0]
    
    # b - How many companies with greater diversity are in the same country
    moreDiv = df[(df['Country Code'] == key_country_code) & (df['Diversity'] > key_row['Diversity'])].shape[0]
    
    # c - Inc/Dec in <params> (here we send the actual values, and can find diff array when graphing)
    feature_cols = {}
    for col in SUBJECT_COLS:
        feature_cols[col] = key_row.filter(like=col)
        
    
    yearly_changes = {}
    for col in SUBJECT_COLS:
        yearly_changes[col] = key_row.filter(like=col).pct_change()
   
    
    # d - how many companies have greater <params> than this company (considering latest year)
    ans = []
    for param in SUBJECT_COLS:
        latest_param_val = key_row.filter(like=param).iloc[-1]
        
        # domestically
        n_greater_val_dom = df[(df['Country Code'] == key_country_code) & 
        (df.filter(like=param).iloc[:, -1] > latest_param_val)].shape[0]
        
        # globally
        n_greater_val_glob = df[(df.filter(like=param).iloc[:, -1] > latest_param_val)].shape[0]

        ans.append({'param':param, 'dom_cnt': n_greater_val_dom, 'glob_cnt': n_greater_val_glob})

    # e
    # Stability criteria:
    # Expense to Revenue ratio
    e_r_ratio = get_expense_to_revenue_ratio(feature_cols['Expense'], feature_cols['Revenue'])
    
    # Growth criteria:
    # CAGR ratio
    cagr_ratio = {}
    for col in feature_cols:
        cagr_ratio[col] = calculate_cagr(feature_cols[col].iloc[0], feature_cols[col].iloc[-1], feature_cols[col].index.size)
   
    ret = {
        "a": freq, 
        "b": moreDiv, 
        "c":yearly_changes,
        "d":ans, 
        "e": {
            "e_r_ratio": e_r_ratio, 
            "cagr_ratio": cagr_ratio
        }
    }
    return ret

