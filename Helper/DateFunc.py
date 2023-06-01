import pandas_market_calendars as mcal
from datetime import datetime, timedelta



def strToDatetime(curDate: str):
    return datetime.strptime(curDate, "%Y-%m-%d")


# 1 for Monday, 7 Sunday
def weekday(curDate: str):
    return strToDatetime(curDate).weekday() + 1


def isMarketOpen(curDate: datetime, longMarketName: str):
    # Create a calendar for the stock market (assuming US market)
    nyse = mcal.get_calendar(longMarketName)

    # Check if the stock market is closed
    return nyse.schedule(start_date=curDate, end_date=curDate).empty


# Checks if the market is open on that date
def isMarketOpenStr(curDate: str, market: str):
    return isMarketOpen(strToDatetime(curDate), market)


# Check for the next market that is not closed or hasn't opened yet
def find_next_market_day(marketName: str, current_datetime: datetime = datetime.now()):
    # Finds stock
    nyse = mcal.get_calendar(marketName)

    # Check if the current day is a valid trading day
    if nyse.valid_days(start_date=current_datetime, end_date=current_datetime).size > 0:
        market_schedule = nyse.schedule(start_date=current_datetime.date(), end_date=current_datetime.date())
        # market_open_time = market_schedule['market_open'].iloc[0].time()
        market_close_time = market_schedule['market_close'].iloc[0].time()
        if current_datetime.time() < market_close_time:
            return current_datetime.date()

    # Find the next market day
    next_market_day = nyse.valid_days(start_date=current_datetime + timedelta(days=1),
                                      end_date=current_datetime + timedelta(days=365)).min().date()

    return next_market_day
