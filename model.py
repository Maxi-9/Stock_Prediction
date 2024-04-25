import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from joblib import dump

from metrics import Metrics
from normalizer import Normalizer
from stocks import Features


# Exceptions:
class ModelNotTrainedError(Exception):
    """Exception raised when trying to use an untrained model."""

    def __init(self):
        super().__init__("Model not trained! Must be trained first.")


class ModelAlreadyTrainedError(Exception):
    """Exception raised when trying to retrain an already trained model. Used for less intelligent models, like linear regression"""

    def __init__(self, model_type: str):
        super().__init__(
            f"Model is already trained. {model_type} model type can't be retrained. Use -o to overwrite model."
        )


class BaseClassError(TypeError):
    """Exception raised when trying to instantiate a class that should be inherited."""

    def __init__(self):
        super().__init__(
            "Commons class should not be instantiated directly, please use one of its inheritors."
        )


class Commons(ABC):
    """
    Base Class for all models and should be inherited.
    Not to be initiated directly!
    """

    # Add mappings here to use in CLI
    model_mapping = {
        # Add other mappings here
    }

    def __init__(self, lookback: int = 30):
        # Sets the version of scheme(stored values)
        self.model_version: float = 1.0
        self.model_type: str = self.get_model_type()

        # Model training settings
        self.lookback: int = lookback
        self.predictOn: Optional[Features] = None
        self.trainOn: [Features] = None
        self.training_stock: [str] = []
        self.is_trained = False
        self.features_version: float = Features.get_version()

        # Select model's features
        self._select_features()

        # Create model with settings
        self.model = self.create_model()

        # Normalize
        self.normalizer = None

    def _select_features(self):
        """
        Overwrite to gain fine feature control. Default: None, PredictOn: Close value
        :return: None
        """
        """
        self.trainOn: [Features] = [
            Features.Open,
            Features.High,
            Features.Low,
            # Features.Close,
            Features.Volume,
            Features.Dividends,
            Features.Splits,
            Features.RSI,
            Features.MACD,
            Features.BB,
            Features.Prev_Close,
        ]

        self.predictOn: Features = Features.Close
        """
        raise NotImplementedError()

    def get_features(self) -> [Features]:
        """

        :return: All the features, meaning any feature it is predicting on or trained on
        """
        return Features.combine(self.trainOn, self.predictOn)

    def save_model(self, file: str, compress_lvl=6):
        """
        Saves model to specific file
        :param compress_lvl: Sets the level of compression for the file.
        :param file: Save location in raw string form
        :return: None
        """
        dump(self, file, compress=("gzip", compress_lvl))

    @abstractmethod
    def _train(self, df: pd.DataFrame):
        """
        :param df: DataFrame with all features
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        """
        :param df: Stock market data with all features
        :return: returns just the prediction column (pred_value column)
        """
        raise NotImplementedError()

    @abstractmethod
    def _predict(self, df: pd.DataFrame) -> float:
        """
        :param df: takes input
        :return: returns just the prediction
        """
        raise NotImplementedError()

    def train(self, df: pd.DataFrame):
        """
        Trains model on df. Warning some models have special constrains, such as Linear model can only be trained on 1 set of data.
        :param df: DataFrame with all features
        :return: None
        """
        # torch compile on arm

        self._train(self._normalize(df))

    # Overwritten for prediction of outputs, Adds column of "pred_value" as close predictions from model

    def batch_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Takes the entire dataset and predicts values from it. Useful for calculate metrics of model.
        :param df: Stock market data with all features
        :return: returns just the prediction column (pred_value column)
        """

        return self._inv_normalize(
            np.concatenate(
                (
                    np.full((self.lookback - 1, 1), np.nan),
                    self._batch_predict(self._normalize(df)),
                )
            ),
            based_on=str(self.predictOn),
        )

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predicts on given values, if model type can only handle 1 input row, it will use the last row as the input.

        :param df: takes input
        :return: returns just the prediction
        """

        return self._inv_normalize_value(
            self._predict(self._normalize(df)), str(self.predictOn)
        )

    def _normalize(self, df: pd.DataFrame, convert: [str] = None):
        if self.normalizer is None:
            if convert is None:
                convert = Features.to_list(self.get_features())

            self.normalizer = Normalizer(df, convert)
            df = self.normalizer.scale(df)
        else:
            df = self.normalizer.scale(df)
        return df

    def _inv_normalize(self, df: np.ndarray, based_on: str):
        if self.normalizer is not None:
            return self.normalizer.inv_normalize_np(df, based_on)
        else:
            raise ValueError(f"Normalizer not set, please train first")

    def _inv_normalize_value(self, value: float, based_on: str):
        if self.normalizer is not None:
            return self.normalizer.inv_normalize_value(value, based_on)
        else:
            raise ValueError(f"Normalizer not set, please train first")

    def calculate_metrics(
        self,
        df: pd.DataFrame,
        range_threshold=0.05,
        initial_capital=10000,
        risk_free_rate=0.05,
        periods_per_year=252,
    ) -> Metrics:
        """
        Takes input df with all the columns (predicted values, and predicted on values) and calculates metrics on it.

        :param df: DataFrame with all columns (features + predicted values)
        :param range_threshold: The threshold for calculating the hit rate
        :param initial_capital: The initial capital for financial calculations
        :param risk_free_rate: Risk-free rate for Sharpe and Sortino ratios
        :param periods_per_year: Trading periods per year for Sharpe and Sortino ratios
        :return: Metrics object
        """
        # unshift shift_close column to get the true close
        df = df.copy()

        df = df.dropna()

        y_pred = df["pred_value"]
        y_true = df[str(self.predictOn)]

        buy_df = Commons.get_buy_rows(df)
        open_vals = buy_df[str(Features.Open)]
        close_vals = buy_df[str(Features.Close)]

        mse = Metrics.calculate_mse(y_true, y_pred)
        r2 = Metrics.calculate_r2(y_true, y_pred)
        mae = Metrics.calculate_mae(y_true, y_pred)
        rmse = Metrics.calculate_rmse(y_true, y_pred)
        cv = Metrics.calculate_cv(y_true)
        mpe = Metrics.calculate_mpe(y_true, y_pred)
        mape = Metrics.calculate_mape(y_true, y_pred)
        smape = Metrics.calculate_smape(y_true, y_pred)
        directional_accuracy = Metrics.calculate_directional_accuracy(y_true, y_pred)

        # Financial metrics use filtered open/close values
        profit_rate = Metrics.calculate_profit_rate(open_vals, close_vals)

        cumulative_return = Metrics.calculate_cumulative_return(
            open_vals, close_vals, initial_capital
        )
        maximum_drawdown = Metrics.calculate_maximum_drawdown(
            open_vals, close_vals, initial_capital
        )
        sharpe_ratio = Metrics.calculate_sharpe_ratio(
            open_vals, close_vals, risk_free_rate, periods_per_year
        )
        sortino_ratio = Metrics.calculate_sortino_ratio(
            open_vals, close_vals, risk_free_rate, periods_per_year
        )

        return Metrics(
            mse=mse,
            r2=r2,
            mae=mae,
            rmse=rmse,
            cv=cv,
            mpe=mpe,
            mape=mape,
            smape=smape,
            directional_accuracy=directional_accuracy,
            profit_rate=profit_rate,
            cumulative_return=cumulative_return,
            maximum_drawdown=maximum_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
        )

    @staticmethod
    def load_from_file(file: str, if_exists=False):
        """
        Load model from file.
        :param file: Raw file location
        :param if_exists: If file doesn't exist, don't throw error if True
        :return: returns the model(of any variant) from specified file
        """
        try:
            return joblib.load(file)
        except FileNotFoundError:
            if if_exists:
                return None
            else:
                raise FileNotFoundError

    @abstractmethod
    def create_model(self):
        """

        :return: A new model
        """
        raise NotImplementedError()

    @staticmethod
    def get_model_type() -> str:
        """
        This is what differences models in files and should just return a string
        :return: model type in string
        """
        # If you created a custom model, don't forget to add your model to Commons.model_mapping
        raise BaseClassError()

    @staticmethod
    def get_buy_rows(df):
        # Filter rows where predicted close value is larger than the open value

        buy_df = df[df["pred_value"] > df[str(Features.Open)]]

        return buy_df


def import_children(directory="Types"):
    models_dir = os.path.join(os.path.dirname(__file__), directory)
    sys.path.insert(0, models_dir)  # Add the directory to sys.path
    for file in os.listdir(models_dir):
        if file.endswith("Model.py"):
            model_name = file[:-3]  # Remove the .py extension
            try:
                module = importlib.import_module(model_name)
                model_class = getattr(module, model_name)
                if hasattr(model_class, "get_model_type"):
                    model_type = model_class.get_model_type()
                    Commons.model_mapping[model_type] = model_class
                else:
                    print(
                        f"Warning: The class {model_name} does not implement 'get_model_type'."
                    )
            except (ImportError, AttributeError) as e:
                print(f"Error importing {model_name}: {e}")
    sys.path.remove(models_dir)  # Remove the directory from sys.path after importing


# Assuming your child classes are in a directory named 'children'
import_children()
