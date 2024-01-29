import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import (
    mean_absolute_error,
)
from sklearn.metrics import mean_squared_error, r2_score

from stocks import Features


# Exceptions:
class ModelNotTrainedError(Exception):
    """Exception raised when trying to use an untrained model."""

    def __init(self):
        super().__init__("Model not trained! Must be trained first.")


class ModelAlreadyTrainedError(Exception):
    """Exception raised when trying to retrain an already trained model."""

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


class Metrics:
    @staticmethod
    def calculate_mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def calculate_r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def calculate_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_cv(y_true):
        return np.std(y_true.values.flatten()) / np.mean(y_true.values.flatten())

    @staticmethod
    def calculate_mpe(y_true, y_pred):
        epsilon = 1e-10  # small constant
        y_true, y_pred = np.nan_to_num(y_true), np.nan_to_num(y_pred)
        return np.nanmean((y_true - y_pred) / (y_true + epsilon))

    @staticmethod
    def calculate_mape(y_true, y_pred):
        epsilon = 1e-10  # small constant
        y_true, y_pred = np.nan_to_num(y_true), np.nan_to_num(y_pred)
        return np.nanmean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    @staticmethod
    def calculate_smape(y_true, y_pred):
        epsilon = 1e-10  # small constant
        y_true, y_pred = np.nan_to_num(y_true), np.nan_to_num(y_pred)
        return (
            np.nanmean(
                2
                * np.abs(y_pred - y_true)
                / (np.abs(y_true) + np.abs(y_pred) + epsilon)
            )
            * 100
        )

    def __init__(
        self,
        mse,
        r2,
        mae=None,
        rmse=None,
        cv=None,
        mpe=None,
        mape=None,
        smape=None,
    ):
        self.mse = mse
        self.r2 = r2
        self.mae = mae
        self.rmse = rmse
        self.cv = cv
        self.mpe = mpe
        self.mape = mape
        self.smape = smape

    def __str__(self):
        metrics = [
            ("MSE (lb)", self.mse),
            ("R2 (hb)", self.r2),
            ("MAE (lb)", self.mae),
            ("RMSE (lb)", self.rmse),
            ("CV (lb)", self.cv),
            ("MPE (lb)", self.mpe),
            ("MAPE (lb)", self.mape),
            ("SMAPE (lb)", self.smape),
        ]

        max_len = max(len(name) for name, _ in metrics)

        lines = []
        for name, value in metrics:
            if value is not None:
                lines.append(f"{name.ljust(max_len)}: {value}")

        return "\n".join(lines)

    def print_metrics(self):
        print(str(self))


# Common Class for model types to inherit to get full program support
class Commons:
    """
    Base Class for all models and should be inherited to gain full functionality and methods of class.
    Not to be initiated directly!
    """

    def __init__(self):
        # Sets the version of scheme(stored values)
        self.model_version: float = 1.0
        self.model_type = self.get_model_type()
        self.model = self.create_model()

        self.predictOn = None
        self.trainOn = None
        self.training_stock = []
        self.is_trained = False
        self.features_version: float = Features.get_version()

        # Select model's features
        self._select_features()

    def _select_features(self):
        """
        Overwrite to gain fine feature control. Default: All training features, PredictOn: Close value
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

    def train(self, df: pd.DataFrame):
        """
        Trains model on df. Warning some models have special constrains, such as Linear model can only be trained on 1 set of data.
        :param df: DataFrame with all features
        :return: None
        """
        raise NotImplementedError()

    # Overwritten for prediction of outputs, Adds column of "pred_value" as close predictions from model
    def test_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the entire dataset and predicts values from it. Adds pred_value column with model's predictions. Should be used to calculate metrics of model.
        :param df: Stock market data with all features
        :return: same df but with pred_value column
        """
        raise NotImplementedError()

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predicts on given values, if model type can only handle 1 input row, it will use the last row as the input.

        :param df: takes input
        :return: returns just the prediction
        """
        raise NotImplementedError()

    def calculate_metrics(self, df: pd.DataFrame) -> Metrics:
        """
        Takes input df with all the columns (predicted values, and predicted on values) and calculates metrics on it

        :param df: DataFrame with all columns (features + predicted values)
        :return: Metrics object
        """

        y_true = df[self.predictOn.columns()]
        y_pred = df["pred_value"]

        mse = Metrics.calculate_mse(y_true, y_pred)
        r2 = Metrics.calculate_r2(y_true, y_pred)
        mae = Metrics.calculate_mae(y_true, y_pred)
        rmse = Metrics.calculate_rmse(y_true, y_pred)
        cv = Metrics.calculate_cv(y_true)
        mpe = Metrics.calculate_mpe(y_true, y_pred)
        mape = Metrics.calculate_mape(y_true, y_pred)
        smape = Metrics.calculate_smape(y_true, y_pred)

        return Metrics(mse, r2, mae, rmse, cv, mpe, mape, smape)

    @staticmethod
    def load_from_file(file: str, if_exists=False):
        """
        Load model from file.
        :param file: Raw file location
        :param if_exists: If file doesn't exist, don't throw error if True
        :return: returns the model(of any variant) from specified file
        """
        try:
            return load(file)
        except FileNotFoundError:
            if if_exists:
                return None
            else:
                raise FileNotFoundError

    def create_model(self):
        """

        :return: A new model
        """
        raise NotImplementedError()

    def get_model_type(self):
        """

        :return: Selected model type
        """
        raise BaseClassError()
