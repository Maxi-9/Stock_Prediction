import torch
import torch.nn as nn
import torch.utils.data
from overrides import overrides

from model import *
from stocks import StockData, Features


class SequentialModel(Commons):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            print("Running on ", self.device)
            self.to(self.device)  # Move the model to the device

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                self.device
            )
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                self.device
            )
            x = x.to(self.device)  # Move the input data to the device
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    def __init__(self):
        # Regular settings
        # self.hidden_size = 64
        # self.num_layers = 2
        # self.learning_rate = 0.001
        # self.num_epochs = 100
        # self.batch_size = 32

        # More capable settings
        self.hidden_size = 512
        self.num_layers = 6
        self.learning_rate = 0.0001
        self.num_epochs = 200
        self.batch_size = 64

        super().__init__()

    @staticmethod
    def get_model_type() -> str:
        return "LSTM"

    @overrides
    def _select_features(self):
        self.trainOn = [
            Features.Open,
            # Features.High,
            # Features.Low,
            Features.RSI,
            Features.MACD,
            Features.BB,
            Features.Prev_Close,
            # Features.Date,
        ]
        self.predictOn = Features.Close

    @overrides
    def create_model(self):
        input_size = len(Features.to_list(self.trainOn))
        output_size = 1
        return self.LSTMModel(
            input_size, self.hidden_size, self.num_layers, output_size
        )

    @overrides
    def _train(self, df: pd.DataFrame):
        x, y = StockData.train_split(df, self.trainOn, self.predictOn)
        x_rolled, y_rolled = StockData.create_rolling_windows(x, y, self.lookback)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_rolled, dtype=torch.float32),
            torch.tensor(y_rolled, dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        try:
            for epoch in range(self.num_epochs):
                for inputs, targets in train_loader:
                    inputs = inputs.to(
                        self.model.device
                    )  # Move input data to the device
                    targets = targets.to(
                        self.model.device
                    )  # Move targets to the device
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}"
                    )
        except KeyboardInterrupt:
            print("Stopped training early")

        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        x_test, y_test = StockData.train_split(df, self.trainOn, self.predictOn)
        x_test_values = x_test.values

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(x_test_values) - self.lookback + 1):
                x_window = x_test_values[i : i + self.lookback]
                x_window = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0)
                output = self.model(x_window)
                predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        return predictions

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        x_pred = df[self.trainOn].values[-self.lookback :]
        x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x_pred)
            prediction = output.cpu().item()

        return prediction


# Uncomment this to add to train.py/test.py/predict.py automatically
Commons.model_mapping["Sequential"] = SequentialModel
