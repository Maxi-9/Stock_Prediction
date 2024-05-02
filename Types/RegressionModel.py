import numpy as np
import pandas as pd
from overrides import overrides
from sklearn.linear_model import LinearRegression

from Tools.stocks import StockData, Features
from model import Commons, ModelNotTrainedError


class RegressionModel(Commons):
    # Creates new model
    def __init__(self):
        super().__init__(lookback=1)

    @staticmethod
    def get_model_type() -> str:
        return "Linear"

    @overrides
    def create_model(self):
        return LinearRegression()

    # Trains Model on given data
    @overrides
    def _train(self, df: pd.DataFrame):
        # Create a new DataFrame with the necessary columns

        x, y = StockData.train_split(df, self.trainOn, self.predictOn)

        # Train the model on the dataset
        self.model.fit(x, y)
        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        x_test, y_test = StockData.train_split(df, self.trainOn, self.predictOn)

        predictions = []
        for i in range(len(x_test) - self.lookback + 1):
            x_window = x_test[i : i + self.lookback]

            # Make a prediction for the next time step
            prediction = self.model.predict(x_window)
            predictions.append(prediction[0])

        return np.array(predictions)

    @overrides
    def _select_features(self):
        self.trainOn: [Features] = [
            Features.Open,
            # Features.High,
            # Features.Low,
            # Features.Close, # Don't Include if predicting on
            # Features.Volume,
            # Features.Dividends,
            # Features.Splits,
            Features.RSI,
            Features.MACD,
            Features.BB,
            Features.Prev_Close,
            # Features.Date,
        ]

        self.predictOn: Features = Features.Close

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        if len(df) < 1:
            raise ValueError("Input DataFrame should have at least one row")
        if self.is_trained is not True:
            raise ModelNotTrainedError()

        x = df[Features.to_list(self.trainOn)].iloc[[-1]]  # Select the last column

        # Predict the target values using the model
        prediction = self.model.predict(x)

        # Get result
        return prediction


# Important, as it adds the model to the CLI
Commons.model_mapping["Linear"] = RegressionModel
