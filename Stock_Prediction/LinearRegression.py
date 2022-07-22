import quandl
import pandas as pd
import numpy as np
from datetime import date

from requests import options
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
#replacement for cross validation
from sklearn.model_selection import train_test_split

def LinearPredection(Options,df):
    df = df[['Close']]
    forecast_out = int(30)

    df['Prediction'] = df[['Close']].shift(-forecast_out)

    X = np.array(df.drop(['Prediction'],1))
    
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
    print("confidence: ", confidence)

    forecast_prediction = clf.predict(X_forecast)
    print(forecast_prediction)
    temp=[]
    for count in forecast_prediction:
        temp.append(count)


    if (options["-graph"]=="Interactable" or options["-graph"]=="Regular"):
        if (options["-graph"]=="Interactable"):
            pass
            #%matplotlib widget
            
        sns.lineplot(data=df)
        sns.set_theme()  # Default seaborn style
        plt.xticks(rotation=30)

        plt.title(f"Closing Stock Prices")
        plt.show()
