import math

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
)
from sklearn.metrics import mean_squared_error, r2_score


class Metrics:
    @classmethod
    def create(
        cls,
        y_true,
        y_pred,
        buy_true,
        buy_pred,
        price_dif,
        risk_free_rate=0.05,
        periods_per_year=252,
        is_number: bool = True,
    ):
        """
        Takes input df with all the columns (predicted values, and predicted on values) and calculates metrics on it.

        :param y_true: The true values
        :param y_pred: The predicted values from the model
        :param buy_true: The true buy values based on if the model had perfect data
        :param buy_pred: The model's buy values based on predicted values
        :param price_dif: The difference between the prices, represents the profit(or loss) if bought.
        :param is_number: If the input y values are floats and can be used for calculations
        :param risk_free_rate: Risk-free rate for Sharpe and Sortino ratios
        :param periods_per_year: Trading periods per year for Sharpe and Sortino ratios
        :return: Metrics object
        """

        directional_accuracy = cls.calculate_directional_accuracy(buy_true, buy_pred)
        profit_rate = cls.calculate_profit_rate(buy_true, buy_pred)
        buy_rate = cls.calculate_buy_rate(buy_pred)
        cumulative_return = cls.calculate_return(price_dif, buy_pred)
        sharpe_ratio = cls.calculate_sharpe_ratio(
            price_dif, buy_pred, risk_free_rate, periods_per_year
        )
        sortino_ratio = cls.calculate_sortino_ratio(
            price_dif, buy_pred, risk_free_rate, periods_per_year
        )

        if is_number:
            mse = cls.calculate_mse(y_true, y_pred)
            r2 = cls.calculate_r2(y_true, y_pred)
            mae = cls.calculate_mae(y_true, y_pred)
            rmse = cls.calculate_rmse(y_true, y_pred)
            cv = cls.calculate_cv(y_true)
            mpe = cls.calculate_mpe(y_true, y_pred)
            mape = cls.calculate_mape(y_true, y_pred)
            smape = cls.calculate_smape(y_true, y_pred)

            return Metrics(
                mse=mse,
                r2=r2,
                mae=mae,
                rmse=rmse,
                cv=cv,
                mpe=mpe,
                mape=mape,
                smape=smape,
                directional_accuracy=directional_accuracy,
                buy_rate=buy_rate,
                profit_rate=profit_rate,
                cumulative_return=cumulative_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
            )
        else:
            return Metrics(
                directional_accuracy=directional_accuracy,
                profit_rate=profit_rate,
                buy_rate=buy_rate,
                cumulative_return=cumulative_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
            )

    def __init__(
        self,
        mse=None,
        r2=None,
        mae=None,
        rmse=None,
        cv=None,
        mpe=None,
        mape=None,
        smape=None,
        directional_accuracy=None,
        buy_rate=None,
        profit_rate=None,
        cumulative_return=None,
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
        self.buy_rate = buy_rate
        self.profit_rate = profit_rate
        self.cumulative_return = cumulative_return
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio

    @staticmethod
    def format_value(value):
        # Split the value into the integer and decimal parts
        integer_part, decimal_part = divmod(value, 1)

        # If the value is purely decimal (no integer part)
        if integer_part == 0:
            # Keep four significant figures
            return f"{value:.4g}"
        else:
            # If the value has an integer part, check for leading zeros in the decimal part
            if decimal_part >= 0.0001:
                # If less than 4 zeros, keep two decimal places
                return f"{integer_part + round(decimal_part, 2):.2f}"
            else:
                # If four or more leading zeros, round down (truncate the decimal part)
                return f"{math.floor(value)}"

    def __str__(self, reduced=False):
        if reduced is True:
            metrics = [
                ("Directional Accuracy (%)", self.directional_accuracy),
                ("Hit Rate (%)", self.profit_rate),
                ("Cumulative Return 1y (x)", self.cumulative_return),
            ]
        else:
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
                ("Buy Rate (%)", self.buy_rate),
                ("Hit Rate (%)", self.profit_rate),
                ("Cumulative Return 1y (x)", self.cumulative_return),
                ("Sharpe Ratio", self.sharpe_ratio),
                ("Sortino Ratio", self.sortino_ratio),
            ]

        max_len = max(len(name) for name, _ in metrics)

        lines = []
        for name, value in metrics:
            if value is not None:
                formatted_value = self.format_value(value)
                lines.append(f"{name.ljust(max_len)}: {formatted_value}")

        return "\n".join(lines)

    def print_metrics(self):
        print(str(self))

    @staticmethod
    def calculate_return(price_dif, buy_pred, initial_capital=1, periods_per_year=252):
        """
        Calculate the cumulative return given price differences and buy predictions.

        :param periods_per_year: Number of days per year
        :param price_dif: Series of price differences (daily returns)
        :param buy_pred: Series of buy predictions (1 for buy, 0 for not buy)
        :param initial_capital: Initial capital for investment
        :return: Final cumulative return
        """

        filtered_values = price_dif[buy_pred]

        # Calculate the cumulative sum of the filtered values
        cumulative_product = filtered_values.cumprod()[-1]

        return cumulative_product * initial_capital / len(price_dif) * periods_per_year

    @staticmethod
    def calculate_sharpe_ratio(
        price_dif, buy_pred, annual_risk_free_rate=0.05, periods_per_year=252
    ):
        # Convert price differences to returns
        returns = price_dif[buy_pred] - 1  # Assuming price_dif values around 1

        # Calculate the risk-free rate for the corresponding period
        period_risk_free_rate = (1 + annual_risk_free_rate) ** (
            1 / periods_per_year
        ) - 1

        # Calculate average return and standard deviation
        avg_return = returns.mean()
        std_dev = returns.std()

        # Handle division by zero case
        if std_dev == 0:
            return None

        # Calculate and return Sharpe ratio
        return (avg_return - period_risk_free_rate) / std_dev

    @staticmethod
    def calculate_sortino_ratio(
        price_dif, buy_pred, annual_risk_free_rate=0.05, periods_per_year=252
    ):
        # Convert price differences to returns
        returns = price_dif[buy_pred] - 1  # Assuming price_dif values around 1

        # Calculate the risk-free rate for the corresponding period
        period_risk_free_rate = (1 + annual_risk_free_rate) ** (
            1 / periods_per_year
        ) - 1

        # Calculate average return
        avg_return = returns.mean()

        # Calculate downside deviation
        downside_returns = returns[returns < period_risk_free_rate]

        if len(downside_returns) > 1:
            downside_deviation = downside_returns.std()
        else:
            downside_deviation = 0

        # Handle no downside deviation case
        if downside_deviation == 0:
            return None

        # Calculate and return Sortino ratio
        return (avg_return - period_risk_free_rate) / downside_deviation

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
    def calculate_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

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

    @staticmethod
    def calculate_directional_accuracy(buy_true, buy_pred):
        return np.sum(buy_true == buy_pred) / len(buy_true) * 100

    @staticmethod
    def calculate_profit_rate(buy_true, buy_pred):
        # Ensure the inputs are numpy arrays

        # Calculate the number of correct buy predictions (true positives)
        true_positives = np.sum((buy_true == buy_pred) & (buy_true == True))

        # Calculate the percentage
        accuracy_percentage = (true_positives / np.sum(buy_pred == True)) * 100

        return accuracy_percentage

    @classmethod
    def calculate_buy_rate(cls, buy_pred):
        return (buy_pred == True).sum() / len(buy_pred) * 100
