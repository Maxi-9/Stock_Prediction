import yfinance as yf

def GetStocks(options):
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period=options["-period"])
    return df