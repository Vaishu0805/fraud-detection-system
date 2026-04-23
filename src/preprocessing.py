import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df = df.drop(['Time'], axis=1)
    return df