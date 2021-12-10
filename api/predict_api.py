import sys
import os
import numpy as np
import pandas as pd
import pickle
from pandas.core.frame import DataFrame 
from sklearn.linear_model import LinearRegression


RSEED = 42

MODEL_PATH = 'api/lr-model.pickle'
DATA_PATH = 'data/GEFCom2014Data/Wind/raw_data_incl_features.csv'
features = ['U10','V10', 'U100', 'V100', 'HOUR', 'MONTH', 'WEEKDAY', 'IS_HOLIDAY', 'WS10', 'WS100', 'WD10', 'WD100', 'U100NORM', 'V100NORM', 'WD100CARD_E', 'WD100CARD_ENE', 'WD100CARD_ESE', 'WD100CARD_N', 'WD100CARD_NE', 'WD100CARD_NNE', 'WD100CARD_NNW', 'WD100CARD_NW', 'WD100CARD_S', 'WD100CARD_SE', 'WD100CARD_SSE', 'WD100CARD_SSW', 'WD100CARD_SW', 'WD100CARD_W', 'WD100CARD_WNW', 'WD100CARD_WSW', 'WD10CARD_E', 'WD10CARD_ENE', 'WD10CARD_ESE', 'WD10CARD_N', 'WD10CARD_NE', 'WD10CARD_NNE', 'WD10CARD_NNW', 'WD10CARD_NW', 'WD10CARD_S', 'WD10CARD_SE', 'WD10CARD_SSE', 'WD10CARD_SSW', 'WD10CARD_SW', 'WD10CARD_W', 'WD10CARD_WNW', 'WD10CARD_WSW']



def load_data_day(data_path=DATA_PATH, day='', zone = -1):
    try:
        data = pd.read_csv(data_path, parse_dates=['TIMESTAMP'])
        data.head()
        data.dropna(inplace=True)
        data = pd.get_dummies(data, columns = ['WD100CARD','WD10CARD'])
        data.drop('TARGETVAR', axis=1, inplace=True)
        ## get only rows of the day in question
        data = data[data['TIMESTAMP'].dt.date.apply(lambda x : str(x)) == day]
        ## for test purposes, take only zone 1
        if int(zone) > 0 or int(zone) < 11:
            data = data[data['ZONEID'] == int(zone)]

        if 'TIMESTAMP' not in features:
            features.append('TIMESTAMP')
        data_wind = data[['ZONEID','TIMESTAMP', 'WS100', 'WD100']]
        data = data[features]
        return data, data_wind
    except OSError as err:
        print(' Cannot load data file.')
        print("OS error: {0}".format(err))
    except BaseException as err:
        print(' Cannot load data file.')
        print(f"Unexpected {err=}, {type(err)=}.")
        raise

def load_model(filename=MODEL_PATH):
    model = LinearRegression()
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except OSError as err:
        print("OS error: {0}".format(err))
    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
    return

def make_prediction(day='', modelpath=MODEL_PATH, zone=-1):
    model = load_model(modelpath)

    # if int(zone) != -1:
    #     data = load_data_day(DATA_PATH, day, zone)
    #     pred = model.predict(data)
    #     pred = [1 if x >1 else 0 if x <0  else x for x in pred] 
    #     pred = [round(x,2) for x in pred]
    #     pred = [[h, x] for h,x in zip(range(0,24), pred)]
    #     pred.insert(0, zone)

    data_wind = None
    df_pred = pd.DataFrame()
    for zone in range(1,11):
        data, data_wind_temp = load_data_day(DATA_PATH, day, zone)
        if zone==1:
            df_pred = data.copy(deep=True)
            data_wind = data_wind_temp.copy(deep=True)
            df_pred.reset_index(inplace=True)
            df_pred = pd.DataFrame(df_pred['TIMESTAMP'])
        data_wind = pd.concat([data_wind, data_wind_temp], axis=0)
        data.drop(['TIMESTAMP'], axis=1, inplace=True)
        temp = model.predict(data)
        temp = [1 if x>1 else 0 if x<0  else x for x in temp] 
        temp = [round(x,2) for x in temp]
        temp = pd.Series(temp, name='Zone '+str(zone))
        df_pred = pd.concat([df_pred, temp], axis=1)

    return df_pred, data_wind

if __name__ == "__main__":
    day='2013-01-01'
    data, _ = load_data_day(DATA_PATH, day)
    pred, data_wind = make_prediction(day)
    #print(f'prediction for day {day}: {pred}')
    print(pred.head())
    print(data_wind)
