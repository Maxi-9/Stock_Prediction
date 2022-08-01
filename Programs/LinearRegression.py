import datetime
import quandl
import pandas as pd
import numpy as np
from datetime import date

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
#replacement for cross validation
from sklearn.model_selection import train_test_split

def LinearPredection(options,df):
    pd.set_option('mode.chained_assignment', None)

    df = df[['Close']]
    forecast_out = options["-forecast"]

    df['Prediction'] = df[['Close']].shift(-forecast_out)
    
    X = np.array(df.drop(['Prediction'],axis=1))
    
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]

    y = np.array(df['Prediction'])
    y = y[:-forecast_out]
    # below sentence has been modified for train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    clf = LinearRegression()
    clf.fit(X_train,y_train)

    confidence = clf.score(X_test, y_test)
    

    forecast_prediction = clf.predict(X_forecast)

    
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day


    for i in forecast_prediction:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

    if (options["-graph"]=="Regular"):          
        sns.lineplot(data=df)
        sns.set_theme()  # Default seaborn style
        plt.xticks(rotation=30)

        plt.title(f"Closing Stock Prices")
        plt.show()

    elif options["-graph"]=="Raw":
        print("confidence: ", confidence)
        print(forecast_prediction)
        
