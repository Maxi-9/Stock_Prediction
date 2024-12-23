import pandas as pd

from TimeSeriesPrediction.features import BaseFeature, Features


class DateFeature(BaseFeature):
    def __init__(self):
        columns = ["Year", "Month", "Day", "Day_Of_Week"]
        super().__init__(columns, is_sensitive=False, uses_data=False)

    def _calculate(self, df) -> dict[str, float]:
        result = {
            "Year": df.year,
            "Month": df.month,
            "Day": df.day,
            "Day_Of_Week": df.dayofweek,  # Monday=0, Sunday=6
        }

        return result


Features.add("Date", DateFeature())


class MovingAverageFeature(BaseFeature):
    def __init__(self, window=30):
        columns = ["MA"]
        super().__init__(columns, is_sensitive=True, uses_data=True)
        self.window = window

    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        # Calculate the moving average for the most recent window
        moving_average = df["Close"].rolling(self.window).mean().iloc[-1]
        return {"MA": moving_average}


Features.add("MA", MovingAverageFeature(window=30))
