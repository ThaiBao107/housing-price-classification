import pandas as pd

def load_data():
    data = pd.read_csv('text.csv')
    data.drop(columns=['city'], inplace=True)
    return data



