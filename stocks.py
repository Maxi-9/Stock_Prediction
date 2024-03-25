from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd
import talib
import yfinance as yf


class StockNotFound(Exception):
    def __init__(self, stock: str):
        super().__init__(f"Stock {stock} not found")


class Features(Enum):
    Open = (["Open"], True)
    High = (["High"], True)
    Low = (["Low"], True)
    Close = (["Close"], True)
    Volume = (["Volume"], False)
    Dividends = (["Dividends"], False)
    Splits = (["Stock Splits"], False)
    RSI = (["RSI"], True)
    MACD = (["MACD"], True)
    BB = (["BB_UPPER", "BB_MIDDLE", "BB_LOWER"], True)
    Prev_Close = (["Prev_Close"], True)
    Date = (["Year", "Month", "Day", "Day_Of_Week"], False)

    def __init__(self, columns, is_normalized):
        self._columns = columns
        self._is_normalized = is_normalized

    def __call__(self, is_normalized=None):
        return FeaturesObject(
            self, is_normalized if is_normalized is not None else self._is_normalized
        )

    def columns(self) -> [str]:
        return self._columns

    def is_normalized(self):
        return self._is_normalized

    def __str__(self):
        return str(self.value[0][0])

    def __iter__(self):
        for column in self.value[0]:
            yield column

    @classmethod
    def combine(cls, list1: list["Features"], feature: "Features"):
        new_list = list1.copy()  # Create a copy of list1
        new_list.append(feature)  # Append the feature to the new list
        return new_list  # Return the new list

    @classmethod
    def get_version(cls) -> float:
        return 1.0

    @staticmethod
    def to_list(features):
        return [
            item
            for feature in features
            for item in (
                Features[feature] if isinstance(feature, str) else feature
            ).columns()
        ]

    @classmethod
    def list_normalized(cls, features):
        return [
            item
            for feature in features
            if (
                Features[feature] if isinstance(feature, str) else feature
            ).is_normalized()
            for item in (
                Features[feature] if isinstance(feature, str) else feature
            ).columns()
        ]


class FeaturesObject:
    def __init__(self, feature: Features, is_normalized: bool):
        self.feature = feature
        self.is_normalized = is_normalized
        self.columns = feature.columns()


class Stock_Data:
    def __init__(self, name: str, period: str, features: [Features]):
        self.name = name
        self.period = period

        # Get Stocks
        df = self._get_raw_stock()

        # Get TA features
        if (
            Features.RSI in features
            or Features.MACD in features
            or Features.BB in features
        ):
            df = self._add_features(df)

        # Add shift
        if Features.Prev_Close in features:
            df = self._add_shift(df)

        # Add Date
        if Features.Date in features:
            df = self._add_date(df)

        # Make sure that just the wanted features are present

        # df = df[Features.to_list(features)]

        df = self._drop_na(df)
        self.df = df

    def _get_raw_stock(self) -> pd.DataFrame:
        ticker = yf.Ticker(
            self.name,
        )

        historical_data = ticker.history(period=self.period)
        if historical_data.empty:
            raise StockNotFound(self.name)

        historical_data.index = historical_data.index.tz_localize(None).normalize()

        return historical_data

    @staticmethod
    def create_rolling_windows(
        x: pd.DataFrame, y: pd.DataFrame, lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates rolling windows from input and target DataFrames.

        Args:
            x (pd.DataFrame): Input DataFrame.
            y (pd.DataFrame): Target DataFrame.
            lookback (int): Lookback period (number of time steps).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and target NumPy arrays with rolling windows.
        """
        # Convert input DataFrame to NumPy array
        x_values = x.values

        # Create rolling windows for input data
        x_windows = []
        for i in range(len(x_values) - lookback + 1):
            x_window = x_values[i : i + lookback]
            x_windows.append(x_window)
        x_windows = np.array(x_windows)

        # Create rolling windows for target data
        y_values = y.values.reshape(
            -1, 1
        )  # Reshape target data to have a single column
        y_windows = [
            y_values[i : i + lookback, 0] for i in range(len(y_values) - lookback + 1)
        ]
        y_windows = np.array(y_windows)

        print(f"Input data shape: {x_windows.shape}")
        return (
            x_windows,
            y_windows[:, -1],
        )  # Return the last value of each target window

    # Add more features
    def _add_features(self, df: pd.DataFrame, window_size: int = 182) -> pd.DataFrame:
        historical_data = df.copy()

        # Initialize an empty list to store the features
        features = []

        # Iterate over the dataset using a rolling window
        for i in range(window_size, len(historical_data)):
            window = historical_data.iloc[i - window_size : i]

            # Check if the window has enough data points
            if len(window) < window_size:
                features.append([float("nan")] * 5)  # Append NaN values
            else:
                # Calculate the technical indicators for the current window
                rsi = talib.RSI(window["Close"])[-1]
                macd, _, _ = talib.MACD(
                    window["Close"], fastperiod=12, slowperiod=26, signalperiod=9
                )
                macd_value = macd[-1]
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    window["Close"], timeperiod=window_size
                )
                bb_upper_value, bb_middle_value, bb_lower_value = (
                    bb_upper[-1],
                    bb_middle[-1],
                    bb_lower[-1],
                )

                # Append the features to the list
                features.append(
                    [rsi, macd_value, bb_upper_value, bb_middle_value, bb_lower_value]
                )

        # Create a DataFrame from the features list and align it with the original data
        features_df = pd.DataFrame(
            features,
            index=historical_data.index[window_size:],
            columns=["RSI", "MACD", "BB_UPPER", "BB_MIDDLE", "BB_LOWER"],
        )

        # Combine the technical indicator features with the original data
        historical_data = pd.concat([historical_data, features_df], axis=1)

        return historical_data

    def _add_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        # Shift the Close price to get the previous day's closing price
        df[str(Features.Prev_Close)] = df["Close"].shift(1)

        return df

    def _add_date(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assume df is your DataFrame and 'date' is your date column

        # Components of the date
        df["year"] = df.index.dt.year
        df["month"] = df.index.dt.month
        df["day"] = df.index.dt.day
        df["day_of_week"] = df.index.dt.dayofweek  # Monday=0, Sunday=6

        return df

    def _drop_na(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop any rows with missing values
        return df.dropna()

    # Train:Test split, higher ratio means more training data
    @staticmethod
    def train_test_split(data, train_ratio=0.8):
        split_index = int(len(data) * train_ratio)
        train = data[:split_index]
        test = data[split_index:]
        return train, test

    # Splits into what is getting predicted on and what it's trying predicting
    @staticmethod
    def train_split(
        df: pd.DataFrame,
        training: [Features],
        prediction: Features = Features.Close,
    ) -> (pd.DataFrame, pd.DataFrame):
        # Assume `data` is your DataFrame containing stock data

        features = df[Features.to_list(training)]  # Features in data

        target = df[prediction.columns()]
        return features, target

    def getTodayStocks(self, stock: str):
        ticker = yf.Ticker(stock)

        # Get the latest information (current day)
        latest_data = ticker.history(period="1d")

        return latest_data


"""
def getMarket(stock: str):
    return yf.Ticker(stock).info['exchange']
"""
