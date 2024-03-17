from keras.models import Sequential
from overrides import overrides
from tensorflow.keras.layers import Dense, LSTM, Dropout

from model import *
from stocks import Stock_Data, Features


class SequentialModel(Commons):
    # Creates new model
    def __init__(self):
        super().__init__()

    @overrides
    def get_model_type(self) -> str:
        return "Sequential"

    @overrides
    def create_model(self):
        # Define the Sequential model architecture
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=(len(self.trainOn), self.lookback),
            )
        )

        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=100))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        self.compile()

    def compile(self):
        self.model.compile(optimizer="adam", loss="mse")

    # Trains Model on given data
    @overrides
    def train(self, df: pd.DataFrame):
        # Prepare input and target data
        x, y = Stock_Data.train_split(df, self.trainOn, self.predictOn)
        x = x.last(self.lookback).values  # Reshape input to 3D

        # Create and train the model
        self.model.fit(
            x, y, epochs=10, batch_size=32, shuffle=False
        )  # Adjust epochs and batch_size
        self.is_trained = True

        # self.compile()

    @overrides
    def test_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Split the data
        x_test, y_test = Stock_Data.train_split(df, self.trainOn, self.predictOn)
        x_test = x_test.values.reshape(
            (-1, len(self.trainOn), 30)
        )  # Reshape input to 3D

        # Use the model to make predictions
        pred = self.model.predict(x_test)

        # Add the predictions to the original DataFrame
        df = df.copy()
        df.loc[:, "pred_value"] = pred.flatten()
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

        x = df[Features.to_list(self.trainOn)].iloc[[-1]].values
        x = x.reshape((1, len(self.trainOn), 30))  # Reshape input to 3D

        # Predict the target values using the model
        prediction = self.model.predict(x)

        # Get result
        return prediction[0, 0]  # Assuming scalar prediction


# Important, as it adds the model to the CLI
Commons.model_mapping["Sequential"] = SequentialModel
