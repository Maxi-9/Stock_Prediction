import importlib.util
import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from joblib import dump, load

from metrics import Metrics
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

    def __init__(self):
        # Sets the version of scheme(stored values)
        self.model_version: float = 1.0
        self.model_type: str = self.get_model_type()
        self.model = self.create_model()

        self.predictOn: Optional[Features] = None
        self.trainOn: [Features] = None
        self.training_stock: [str] = []
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

    @abstractmethod
    def train(self, df: pd.DataFrame):
        """
        Trains model on df. Warning some models have special constrains, such as Linear model can only be trained on 1 set of data.
        :param df: DataFrame with all features
        :return: None
        """
        raise NotImplementedError()

    # Overwritten for prediction of outputs, Adds column of "pred_value" as close predictions from model
    @abstractmethod
    def test_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the entire dataset and predicts values from it. Adds pred_value column with model's predictions. Should be used to calculate metrics of model.
        :param df: Stock market data with all features
        :return: same df but with pred_value column
        """
        raise NotImplementedError()

    @abstractmethod
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

    @abstractmethod
    def create_model(self):
        """

        :return: A new model
        """
        raise NotImplementedError()

    @abstractmethod
    def get_model_type(self) -> str:
        """
        This is what differences models in files and should just return a string
        :return: model type in string
        """
        # If you created a custom model, don't forget to add your model to Commons.model_mapping
        raise BaseClassError()


def import_children(directory="Types"):
    for filename in os.listdir(directory):
        if filename.endswith(".py") and not filename.startswith("__"):
            # Construct full path to file
            file_path = os.path.join(directory, filename)
            # Create a module name based on the file name
            module_name = os.path.splitext(filename)[0]
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


# Assuming your child classes are in a directory named 'children'
import_children()
