import datetime

import pandas
from Helper.DateFunc import find_next_market_day


# Fixes the data so that my methods can use them
def fix(df: pandas.DataFrame, marketName: str, todayDate: datetime = datetime.datetime.now(), forceSameDay = False):
    # find the next market day
    # Specify the target date as a pandas.Timestamp, ignoring the time component
    current_date = pandas.Timestamp(find_next_market_day(marketName, current_datetime=todayDate))



    if pandas.Timestamp(current_date) in df.index:
        # Case: Market has opened but not closed yet
        # More accurate

        # Get the row for the target date, remove the 'close' value
        df.loc[df.index.normalize() == current_date, 'Close'] = None

        # Remove all rows after the target date
        df = df[df.index <= current_date]
    # Check if current_date is already present in the DataFrame
    else:
        if forceSameDay:
            print("Not accurate data")
            raise
        # Create a new row with all NaN values and the custom index
        new_row = pandas.Series([float('nan')] * len(df.columns), index=df.columns, name=current_date)

        # Set the 'open' and 'close' values of the new row to the last close value
        last_close = df['Close'].iloc[-1]
        new_row['Open'] = last_close
        new_row['Close'] = last_close

        df = df._append(new_row, ignore_index=False)
        print(df)
    # Fill missing values in the 'Close' column with the last available value
    df['Close'] = df['Close'].fillna(method='ffill')

    return df
