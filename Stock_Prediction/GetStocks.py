import yfinance as yf

def GetStocks(Stock,period="max"):
    ticker = yf.Ticker(Stock)
    df = ticker.history(period=period)
    return df