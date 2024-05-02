from overrides import overrides

from model import *
from stocks import StockData, Features


class SequentialModel(Commons):
    def __init__(self):
        super().__init__()
        self.lookback = 30  # Set lookback

    @staticmethod
    def get_model_type() -> str:
        return "Example"

    @overrides
    def _select_features(self):
        """
        Select the features that will be used in the model (also how many columns will be in df)
        :return:
        """
        self.trainOn = [
            Features.Open,
            Features.High,
            Features.Low,
            Features.RSI,
            Features.MACD,
            Features.BB,
            Features.Prev_Close,
            Features.Date,
        ]
        self.predictOn = Features.Close

    @overrides
    def create_model(self):
        return  # Create instance of model here

    @overrides
    def _train(self, df: pd.DataFrame):
        x, y = StockData.train_split(df, self.trainOn, self.predictOn)
        x_rolled, y_rolled = StockData.create_rolling_windows(x, y, self.lookback)
        # x_rolled has shape (n_samples, lookback, n_features) with features being len(self.trainOn)
        # y_rolled has shape (n_samples,) (978,)
        try:
            pass
            # main training loop
        except KeyboardInterrupt:
            print("Stopped training early")

        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Takes the entire dataset and predicts values from it. Useful for calculate metrics of model.
        :param df: Dataset with features from trainOn without lookback
        :return: returns just the prediction column (pred_value column) with the date index
        """
        x_test, y_test = StockData.train_split(df, self.trainOn, self.predictOn)
        x_rolled, y_rolled = StockData.create_rolling_windows(
            x_test, y_test, self.lookback
        )
        pass

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        """
        Predicts on given values, if model type can only handle 1 input row, it will use the last row as the input.
        :param df: pred input with features from trianOn with lookback
        :return: single float prediction
        """
        pass


# Uncomment this to add to train.py/test.py/predict.py automatically
# Commons.model_mapping["Sequential"] = SequentialModel
