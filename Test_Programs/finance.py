from tokenize import String
from matplotlib import ticker
import yfinance as yf
import numpy as np
#define the ticker symbol
tickerSymbol = 'MSFT'

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2022-4-14')

#see your data
print(tickerDf)

closeValues=[]
dates=[]

    
closeValues=tickerDf["Close"].values.tolist()

for count in tickerDf.index:
    dates.append(str(count))
print(dates)

