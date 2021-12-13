# Import Libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import pickle

features = ['ZONEID', 'U10', 'V10', 'U100', 'V100', 'HOUR', 'MONTH',
       'WEEKDAY', 'WS10', 'WS100', 'WD10', 'WD100', 'U100NORM', 'V100NORM',
       'WD100CARD_ENE', 'WD100CARD_ESE', 'WD100CARD_N', 'WD100CARD_NE',
       'WD100CARD_NNE', 'WD100CARD_NNW', 'WD100CARD_NW', 'WD100CARD_S',
       'WD100CARD_SE', 'WD100CARD_SSE', 'WD100CARD_SSW', 'WD100CARD_SW',
       'WD100CARD_W', 'WD100CARD_WNW', 'WD100CARD_WSW', 'WD10CARD_ENE',
       'WD10CARD_ESE', 'WD10CARD_N', 'WD10CARD_NE', 'WD10CARD_NNE',
       'WD10CARD_NNW', 'WD10CARD_NW', 'WD10CARD_S', 'WD10CARD_SE',
       'WD10CARD_SSE', 'WD10CARD_SSW', 'WD10CARD_SW', 'WD10CARD_W',
       'WD10CARD_WNW', 'WD10CARD_WSW']

# Add columns for cardinal wind directions
def degrees_to_cardinal(d):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((d + 11.25)/22.5)
    return dirs[ix % 16]

# Add columns for wind direction at the two different heights
def uv_to_winddir(u,v):
    return (180 + 180 / np.pi * np.arctan2(u,v)) % 360

def add_missing_dummy_columns(df):
    missing = [ii for ii in features if ii not in df.columns]
    df_zero = pd.DataFrame(np.zeros((df.shape[0], len(missing))),
                columns=missing)
    return pd.concat([df, df_zero], axis=1)



def create_features(df):
    # date and time
    df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP)
    df['HOUR'] = df.TIMESTAMP.dt.hour
    df['MONTH'] = df.TIMESTAMP.dt.month
    df['WEEKDAY'] = df.TIMESTAMP.dt.weekday
    df.drop('TIMESTAMP', axis=1, inplace=True)
    
    # wind direction
    df.eval('WS10 = (U10 ** 2 + V10 ** 2) ** 0.5', inplace=True)
    df.eval('WS100 = (U100 ** 2 + V100 ** 2) ** 0.5', inplace=True)
    df['WD10'] = uv_to_winddir(df.U10, df.V10)
    df['WD100'] = uv_to_winddir(df.U100, df.V100)
    df['WD100CARD'] = df.WD100.apply(lambda x: degrees_to_cardinal(x))
    df['WD10CARD'] = df.WD10.apply(lambda x: degrees_to_cardinal(x))
    
    # Add columns for normed wind vector components (normed by ws)
    df.eval('U100NORM = U100 / WS100', inplace=True)
    df.eval('V100NORM = V100 / WS100', inplace=True)
    
    df = pd.get_dummies(df, columns = ['WD100CARD','WD10CARD'], drop_first=True)
    df = add_missing_dummy_columns(df)
    df = df[features]
    return df


def load_model(model_path):
    model = RandomForestRegressor()
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except OSError as err:
        print("OS error: {0}".format(err))
    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
    return model

def make_prediction(df, model):
    pred = model.predict(df)
    return pred


if __name__ == "__main__":
    data_path = 'raw_data_sin_features_one_day.csv'
    df = pd.read_csv(data_path)
    model_path = '3random_forest_without_holiday.pickle'
    df_feat = create_features(df)

    model = load_model(model_path)
    pred = make_prediction(df_feat, model)
    print('df_pred.shape ',pred.shape)
    print(pred)
    print('########')
    df_pred = df_feat[['ZONEID', 'HOUR']]
    pred = pd.Series(pred)
    df_pred = pd.concat([df_pred, pred], axis=1)
    df_pred.columns = ['ZONEID', 'HOUR', 'TARGETVAR']
    print(df_pred.shape)
    print(df_pred.head())

    df_wide =  df_pred.pivot(index='HOUR', columns='ZONEID', values='TARGETVAR')
    print(df_wide.head())