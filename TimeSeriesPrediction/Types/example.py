from overrides import overrides

from TimeSeriesPrediction.model import *


class SequentialModel(Commons):
    def __init__(self):
        feat = [
            Features.Open,
            Features.BB,
            Features.RSI,
            Features.Date,
            Features.MA,
            Features.MACD,
        ]
        super().__init__(
            None, "Example", Features(feat, Features.Close), lookback=1
        )  # Set lookback to 1 to disable
        # Make sure to add model to index(see bottom of file)

    @overrides
    def use_seed(self, seed: int | None = None):
        super(seed)
        pass  # Use library specific seed setter here

    @overrides
    def _train(self, df: pd.DataFrame):
        x, y = Data.train_split(
            df, self.features.train_cols(), self.features.predict_on
        )
        x_rolled, y_rolled = Data.create_rolling_windows(x, y, self.lookback)
        # x_rolled has shape (n_samples, lookback, n_features) with features being len(self.trainOn)
        # y_rolled has shape (n_samples,) (978,)
        try:
            pass
            # main training loop
        except KeyboardInterrupt:
            print("Stopped training early")

        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        """
        Takes the entire dataset and predicts values from it. Useful for calculate metrics of model.
        :param df: Dataset with features from trainOn without lookback
        :return: returns just the prediction column (pred_value column) with the date index
        """
        x_test, y_test = Data.train_split(
            df, self.features.train_cols(), self.features.predict_on
        )
        x_rolled, y_rolled = Data.create_rolling_windows(x_test, y_test, self.lookback)

        pass

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        """
        Predicts on given values, if model type can only handle 1 input row, it will use the last row as the input.
        :param df: pred input with features from trianOn with lookback
        :return: single float prediction
        """
        pass


# Uncomment this to add to train.py, test.py, and predict.py automatically
# Commons.model_mapping["Sequential"] = SequentialModel
