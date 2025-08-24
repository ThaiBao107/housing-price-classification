import pandas as pd

def load_data():
    data = pd.read_csv('Clean_HCM (1).csv')
    data.drop(columns=['city'], inplace=True)
    return data



