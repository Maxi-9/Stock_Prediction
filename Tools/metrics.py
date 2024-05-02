import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
)
from sklearn.metrics import mean_squared_error, r2_score


class Metrics:
    def __init__(
        self,
        mse,
        r2,
        mae=None,
        rmse=None,
        cv=None,
        mpe=None,
        mape=None,
        smape=None,
        directional_accuracy=None,
        profit_rate=None,
        cumulative_return=None,
        maximum_drawdown=None,
        sharpe_ratio=None,
        sortino_ratio=None,
    ):
        self.mse = mse
        self.r2 = r2
        self.mae = mae
        self.rmse = rmse
        self.cv = cv
        self.mpe = mpe
        self.mape = mape
        self.smape = smape
        self.directional_accuracy = directional_accuracy
        self.hit_rate = profit_rate
        self.cumulative_return = cumulative_return
        self.maximum_drawdown = maximum_drawdown
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio

    def __str__(self):
        metrics = [
            ("MSE (lb)", self.mse),
            ("R2 (hb)", self.r2),
            ("MAE (lb)", self.mae),
            ("RMSE (lb)", self.rmse),
            ("CV (lb)", self.cv),
            ("MPE (lb)", self.mpe),
            ("MAPE (lb)", self.mape),
            ("SMAPE (lb)", self.smape),
            ("Directional Accuracy (%)", self.directional_accuracy),
            ("Hit Rate (%)", self.hit_rate),
            ("Cumulative Return", self.cumulative_return),
            ("Maximum Drawdown", self.maximum_drawdown),
            ("Sharpe Ratio", self.sharpe_ratio),
            ("Sortino Ratio", self.sortino_ratio),
        ]

        max_len = max(len(name) for name, _ in metrics)

        lines = []
        for name, value in metrics:
            if value is not None:
                lines.append(f"{name.ljust(max_len)}: {value}")

        return "\n".join(lines)

    def print_metrics(self):
        print(str(self))

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        return np.sqrt(((y_true - y_pred) ** 2).mean())

    @staticmethod
    def calculate_directional_accuracy(y_true, y_pred):
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same number of elements")

        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)

        if y_true_diff.shape != y_pred_diff.shape:
            raise ValueError(
                "Differences in prediction and true values have different shapes"
            )
        return (np.sign(y_true_diff) == np.sign(y_pred_diff)).mean() * 100

    # @staticmethod
    # def calculate_hit_rate(y_true, y_pred, range_threshold):
    #     within_range = np.abs(y_true - y_pred) <= range_threshold
    #     return within_range.mean() * 100

    @staticmethod
    def calculate_profit_rate(open_vals, close_vals):
        profitable_trades = close_vals > open_vals

        profit_rate = np.mean(profitable_trades) * 100

        return profit_rate

    @staticmethod
    def calculate_cumulative_return(open_vals, close_vals, initial_capital=10000):
        returns = close_vals / open_vals - 1
        cumulative_return = initial_capital * np.prod(1 + returns)
        return cumulative_return

    @staticmethod
    def calculate_maximum_drawdown(open_vals, close_vals, initial_capital=10000):
        portfolio_values = initial_capital * (1 + (close_vals / open_vals - 1).cumsum())
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    @staticmethod
    def calculate_sharpe_ratio(
        open_vals, close_vals, risk_free_rate=0.05, periods_per_year=252
    ):
        """
        Calculates the sharpe ratio.
        :param open_vals: Filtered open values(based on buy method)
        :param close_vals: Filtered close values
        :param risk_free_rate:
        :param periods_per_year:
        :return:
        """
        returns = close_vals / open_vals - 1
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        )
        return sharpe_ratio

    @staticmethod
    def calculate_sortino_ratio(
        open_vals, close_vals, risk_free_rate=0.05, periods_per_year=252
    ):
        returns = close_vals / open_vals - 1
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = np.minimum(excess_returns, 0)
        sortino_ratio = (
            excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
        )
        return sortino_ratio

    @staticmethod
    def calculate_mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def calculate_r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def calculate_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_cv(y_true):
        return np.std(y_true.values.flatten()) / np.mean(y_true.values.flatten())

    @staticmethod
    def calculate_mpe(y_true, y_pred):
        epsilon = 1e-10  # small constant
        y_true, y_pred = np.nan_to_num(y_true), np.nan_to_num(y_pred)
        return np.nanmean((y_true - y_pred) / (y_true + epsilon))

    @staticmethod
    def calculate_mape(y_true, y_pred):
        epsilon = 1e-10  # small constant
        y_true, y_pred = np.nan_to_num(y_true), np.nan_to_num(y_pred)
        return np.nanmean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    @staticmethod
    def calculate_smape(y_true, y_pred):
        epsilon = 1e-10  # small constant
        y_true, y_pred = np.nan_to_num(y_true), np.nan_to_num(y_pred)
        return (
            np.nanmean(
                2
                * np.abs(y_pred - y_true)
                / (np.abs(y_true) + np.abs(y_pred) + epsilon)
            )
            * 100
        )
