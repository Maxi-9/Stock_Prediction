import yfinance as yf

def GetStocks(options):
    ticker = yf.Ticker(options["-stock"])
    df = ticker.history(period=options["-period"])
    return df