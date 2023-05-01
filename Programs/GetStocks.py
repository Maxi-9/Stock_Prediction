import yfinance as yf
import Settings

def GetStocks():
    ticker = yf.Ticker(Settings.Stock)
    df = ticker.history(period=Settings.Period)
    return df