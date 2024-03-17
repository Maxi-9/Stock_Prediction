from enum import Enum

import numpy as np
import pandas as pd
import talib
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


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


class Normalizer:
    # Needs: Data Frame and the columns that you want to be normalized
    # Use: Scalars only hold 1 min/max value, this holds it for all given columns

    def __init__(self, df: pd.DataFrame, scale_cols: [str]):
        self.scalars = {}
        for col in scale_cols:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalars[col] = scaler

        self.df = df

    def get_scaled(self) -> pd.DataFrame:
        return self.df

    def inv_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col, scaler in self.scalars.items():
            if col in data.columns:
                data[col] = scaler.inverse_transform(data[[col]])
        return data

    def inv_normalize_col(
        self, data: pd.DataFrame, convert: str, based_on: str
    ) -> pd.DataFrame:
        data = data.copy()
        scaler = self.scalars.get(based_on)
        if scaler is not None:
            data[convert] = scaler.inverse_transform(data[[convert]])
            return data
        else:
            raise ValueError(f"No scaler found for column: {based_on}")

    def inv_normalize_value(self, value: float, based_on: str):
        scaler = self.scalars.get(based_on)
        if scaler is not None:
            # Reshape your input to match the original data shape
            value = np.array(value).reshape(-1, 1)
            inv_normalized = scaler.inverse_transform(value)[0]
            return inv_normalized
        else:
            raise ValueError(f"No scaler found for column: {based_on}")


class Stock_Data:
    def __init__(self, stock: str, period: str, features: [Features], normalized=True):
        self.stock = stock
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

        # Normalize
        self.normal = None
        if normalized:
            # Save the normal values for later reversal
            self.normal = Normalizer(df, Features.list_normalized(features))
            df = self.normal.get_scaled()

        # Add Date
        if Features.Date in features:
            df = self._add_date(df)

        #
        df = self._drop_na(df)
        self.df = df

    def _get_raw_stock(self) -> pd.DataFrame:
        ticker = yf.Ticker(
            self.stock,
        )

        historical_data = ticker.history(period=self.period)
        if historical_data.empty:
            raise StockNotFound(self.stock)

        historical_data.index = historical_data.index.tz_localize(None).normalize()

        return historical_data

    # Add more features
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        historical_data = df

        # Calculate technical indicators
        historical_data["RSI"] = talib.RSI(historical_data["Close"])
        historical_data["MACD"], _, _ = talib.MACD(historical_data["Close"])
        (
            historical_data["BB_UPPER"],
            historical_data["BB_MIDDLE"],
            historical_data["BB_LOWER"],
        ) = talib.BBANDS(historical_data["Close"])

        return historical_data

    def _add_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        # Shift the Close price to get the previous day's closing price
        df["Prev_Close"] = df["Close"].shift(1)

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

    def inv_normalize_col(self, df: pd.DataFrame, convert: str, based_on: str):
        """

        :param df: Takes normalized df from self.df
        :param convert: column you want to inverse normalize
        :param based_on: converts the selected column into full scale based on this original column
        :return
        """
        if self.normal is not None:
            return self.normal.inv_normalize_col(df, convert, based_on)
        else:
            return df

    def inv_normalize(self, df: pd.DataFrame):
        if self.normal is not None:
            return self.normal.inv_normalize(df)
        else:
            return df

    def inv_normalize_value(self, value, based_on):
        if self.normal is not None:
            return self.normal.inv_normalize_value(value, based_on)
        else:
            return value

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
        look_back: int = 1,
    ) -> (pd.DataFrame, pd.DataFrame):
        # Assume `data` is your DataFrame containing stock data
        features = df[Features.to_list(training)]  # Features in data
        Stock_Data.look_back(df, look_back)
        target = df[prediction.columns()]
        return features, target

    @staticmethod
    def look_back(df: pd.DataFrame, look_back: int = 1) -> pd.DataFrame:
        return df.tail(look_back)

    def getTodayStocks(self, stock: str):
        ticker = yf.Ticker(stock)

        # Get the latest information (current day)
        latest_data = ticker.history(period="1d")

        return latest_data


"""
def getMarket(stock: str):
    return yf.Ticker(stock).info['exchange']
"""
