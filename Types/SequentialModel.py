from keras import Input
from keras.models import Sequential
from overrides import overrides
from tensorflow.keras.layers import Dense, LSTM, Dropout

from model import *
from stocks import Stock_Data, Features


class SequentialModel(Commons):
    # Creates new model
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_model_type() -> str:
        return "Sequential"

    @overrides
    def create_model(self):
        # Define the Sequential model architecture
        model = Sequential()
        model.add(Input(shape=(self.lookback, len(Features.to_list(self.trainOn)))))

        # Add the LSTM layers
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=100))
        model.add(Dropout(0.2))

        # Add the output Dense layer
        model.add(Dense(units=1))

        model.compile(optimizer="adam", loss="mse")

        return model

    # Trains Model on given data
    @overrides
    def _train(self, df: pd.DataFrame):
        """
        Trains the model using the provided data.

        Args:
            df (pd.DataFrame): DataFrame with all features

        Returns:
            None
        """

        # Prepare input and target data
        x, y = Stock_Data.train_split(df, self.trainOn, self.predictOn)
        print(x)
        # Create rolling windows
        x_rolled, y_rolled = Stock_Data.create_rolling_windows(x, y, self.lookback)

        # Train the model using the rolling windows
        self.model.fit(x_rolled, y_rolled, epochs=10, batch_size=32, shuffle=False)

        # Mark the model as trained
        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.ndarray:
        # Split the data
        x_test, y_test = Stock_Data.train_split(df, self.trainOn, self.predictOn)

        # Reshape input data
        x_rolled, y_rolled = Stock_Data.create_rolling_windows(
            x_test, y_test, self.lookback
        )

        # Use the model to make predictions
        predictions = self.model.predict(x_rolled, batch_size=32)

        # Return predictions with padding to original length
        return np.concatenate(
            (np.repeat(predictions[:1, :], self.lookback - 1, axis=0), predictions),
            axis=0,
        )

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
    def _predict(self, df: pd.DataFrame) -> float:
        if len(df) < 1:
            raise ValueError("Input DataFrame should have at least one row")
        if self.is_trained is not True:
            raise ModelNotTrainedError()

        x = df[Features.to_list(self.trainOn)].iloc[[-1]].values
        x = x.reshape((1, len(self.trainOn), 30))  # Reshape input to 3D

        # Predict the target values using the model
        prediction = self.model.predict(x)

        # Get result
        return prediction[0, 0]  # Assuming scalar prediction


# Important, as it adds the model to the CLI
Commons.model_mapping["Sequential"] = SequentialModel
