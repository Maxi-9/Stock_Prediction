import torch
import torch.nn as nn
from overrides import overrides

from model import *
from stocks import Stock_Data, Features


class Vanilla_SequentialModel(Commons):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm1 = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.dropout1 = nn.Dropout(0.1)
            self.lstm2 = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.dropout2 = nn.Dropout(0.1)
            self.lstm3 = nn.LSTM(
                hidden_size, hidden_size, num_layers=1, batch_first=True
            )
            self.dropout3 = nn.Dropout(0.1)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            out, _ = self.lstm3(out)
            out = self.dropout3(out)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_model_type() -> str:
        return "Sequential"

    @overrides
    def _select_features(self):
        self.trainOn = [
            Features.Open,
            # Features.High,
            # Features.Low,
            # Features.RSI,
            # Features.MACD,
            # Features.BB,
            # Features.Prev_Close,
            Features.Date,
        ]
        self.predictOn = Features.Close

    @overrides
    def create_model(self):
        return self.LSTMModel(len(Features.to_list(self.trainOn)), 256)

    @overrides
    def _train(self, df: pd.DataFrame):
        x, y = Stock_Data.train_split(df, self.trainOn, self.predictOn)
        x_rolled, y_rolled = Stock_Data.create_rolling_windows(x, y, self.lookback)
        # print(x_rolled.shape, y_rolled.shape)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        total_batches = 64  # len(x_rolled) // 64 + (len(x_rolled) % 64 != 0)
        total_eophs = 15
        print("Total Batches:", total_batches)
        try:
            for epoch in range(total_eophs):
                running_loss = 0.0
                for batch_idx in range(0, len(x_rolled), 64):
                    batch_x = torch.tensor(
                        x_rolled[batch_idx : batch_idx + 64], dtype=torch.float32
                    )
                    batch_y = torch.tensor(
                        y_rolled[batch_idx : batch_idx + 64], dtype=torch.float32
                    )
                    batch_y = batch_y.unsqueeze(
                        -1
                    )  # Add a new dimension to match the input tensor shape

                    optimizer.zero_grad()
                    outputs = self.model.forward(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # Print progress
                    current_batch = batch_idx // 64 + 1

                    print(
                        f"\rEpoch [{epoch + 1}/{total_eophs}], Batch [{current_batch+1}/{total_batches}], Loss: {running_loss / current_batch:.6f}",
                        end="",
                    )
                    if current_batch + 1 == total_batches:
                        print("")

            # print(
            #     f"Epoch [{epoch + 1}/{20}] completed, Average Loss: {running_loss / total_batches:.6f}"
            # )
        except KeyboardInterrupt:
            print("Stopped training early")

        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        x_test, y_test = Stock_Data.train_split(df, self.trainOn, self.predictOn)
        x_rolled, y_rolled = Stock_Data.create_rolling_windows(
            x_test, y_test, self.lookback
        )

        predictions = []
        with torch.no_grad():
            for batch_idx in range(0, len(x_rolled), 64):
                batch_x = torch.tensor(
                    x_rolled[batch_idx : batch_idx + 64], dtype=torch.float32
                )
                outputs = self.model(batch_x)  # Call the model directly
                predictions.extend(
                    outputs.detach().numpy()
                )  # Detach the outputs before converting to numpy

        return predictions

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        if len(df) < 1:
            raise ValueError("Input DataFrame should have at least one row")
        if not self.is_trained:
            raise ModelNotTrainedError()

        x = df[Features.to_list(self.trainOn)].iloc[-1].values
        x = x.reshape((1, len(self.trainOn), self.lookback))
        x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            prediction = self.model.forward(x)  # Call the forward method explicitly

        return prediction.item()


Commons.model_mapping["Vanilla_Sequential"] = Vanilla_SequentialModel
