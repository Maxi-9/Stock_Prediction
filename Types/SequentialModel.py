import torch
import torch.nn as nn
import torch.utils.data
from overrides import overrides

from model import *
from stocks import Stock_Data, Features


class SequentialModel(Commons):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    def __init__(self):
        self.hidden_size = 64
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 32

        super().__init__()

    @staticmethod
    def get_model_type() -> str:
        return "LSTM"

    @overrides
    def _select_features(self):
        self.trainOn = [
            Features.Open,
            Features.High,
            Features.Low,
            # Features.RSI,
            # Features.MACD,
            # Features.BB,
            Features.Prev_Close,
            Features.Date,
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
        x, y = Stock_Data.train_split(df, self.trainOn, self.predictOn)
        x_rolled, y_rolled = Stock_Data.create_rolling_windows(x, y, self.lookback)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_rolled, dtype=torch.float32),
            torch.tensor(y_rolled, dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        model = self.create_model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        try:
            for epoch in range(self.num_epochs):
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}"
                    )
        except KeyboardInterrupt:
            print("Stopped training early")

        self.model = model
        self.is_trained = True

    @overrides
    def _batch_predict(self, df: pd.DataFrame) -> np.array:
        x_test, y_test = Stock_Data.train_split(df, self.trainOn, self.predictOn)
        x_rolled, y_rolled = Stock_Data.create_rolling_windows(
            x_test, y_test, self.lookback
        )

        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_rolled, dtype=torch.float32)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size
        )

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0]
                outputs = self.model(inputs)
                predictions.extend(outputs.numpy())

        return np.array(predictions).reshape(-1, 1)

    @overrides
    def _predict(self, df: pd.DataFrame) -> float:
        x_pred = df[self.trainOn].values[-self.lookback :]
        x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x_pred)
            prediction = output.item()

        return prediction


# Uncomment this to add to train.py/test.py/predict.py automatically
Commons.model_mapping["Sequential"] = SequentialModel
