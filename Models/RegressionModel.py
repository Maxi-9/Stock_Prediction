import pandas as pd
from overrides import overrides
from sklearn.linear_model import LinearRegression

from model import Commons, ModelNotTrainedError, ModelAlreadyTrainedError
from stocks import Stock_Data, Features


class RegressionModel(Commons):
    # Creates new model
    def __init__(self):
        super().__init__()

    @overrides
    def get_model_type(self):
        return "Linear"

    @overrides
    def create_model(self):
        return LinearRegression()

    # Trains Model on given data
    @overrides
    def train(self, df: pd.DataFrame):
        if self.training_stock:
            raise ModelAlreadyTrainedError(self.get_model_type())
        # Create a new DataFrame with the necessary columns

        x, y = Stock_Data.train_split(df, self.trainOn, self.predictOn)

        # Train the model on the dataset
        self.model.fit(x, y)
        self.is_trained = True

    @overrides
    def test_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Split the data
        x_test, y_test = Stock_Data.train_split(df, self.trainOn, self.predictOn)

        # Use the model to make predictions

        pred = self.model.predict(x_test)

        # Add the predictions to the original DataFrame
        df = df.copy()
        df.loc[:, "pred_value"] = pred

        return df

    @overrides
    def _select_features(self):
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

    @overrides
    def predict(self, df: pd.DataFrame) -> float:
        if len(df) < 1:
            raise ValueError("Input DataFrame should have at least one row")
        if self.is_trained is not True:
            raise ModelNotTrainedError()

        x = df[Features.to_list(self.trainOn)].iloc[[-1]]  # Select the last column

        # Predict the target values using the model
        prediction = self.model.predict(x)

        # Get result
        return prediction
