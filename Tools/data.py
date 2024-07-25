from typing import Tuple

import numpy as np
import pandas as pd


class Data:
    @staticmethod
    def drop_na(df: pd.DataFrame):
        df.dropna(inplace=True)

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
        training: [str],
        prediction,
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Splits into what is getting predicted on and what it's trying predicting
        :param df: Raw dataframe with all data/features
        :param training: Columns to train on
        :param prediction: Column to predict(Needs to be BaseFeature object)
        :return: training data, prediction data
        """
        # Assume `data` is your DataFrame containing stock data

        features = df[training]  # Features in data

        target = df[prediction.true_col()]
        return features, target

    @staticmethod
    def sanitize_name(name):
        # Define a set of illegal characters for Excel sheet titles
        illegal_chars = ["\\", "/", "*", "[", "]", ":", "?"]

        for char in illegal_chars:
            name = name.replace(char, "_")
        return name
