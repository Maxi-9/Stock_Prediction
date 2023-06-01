import yfinance as yf


def getStocks(stock: str, period: str):
    ticker = yf.Ticker(stock)

    # Get the latest information (current day)
    # latest_data = ticker.history(period='1d')

    # Get historical data using Settings.Period
    historical_data = ticker.history(period=period)

    # Remove the time component from the index
    historical_data.index = historical_data.index.tz_localize(None).normalize()

    # Remove the time component from the index
    # latest_data.index = latest_data.index.tz_localize(None).normalize()

    # Concatenate the latest data and historical data
    # df = pd.concat([latest_data, historical_data])

    return historical_data

def getTodayStocks(stock: str):
    ticker = yf.Ticker(stock)

    # Get the latest information (current day)
    latest_data = ticker.history(period='1d')

    return latest_data

def getMarket(stock: str):
    return yf.Ticker(stock).info['exchange']
