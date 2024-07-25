# Analysis
import numpy as np
import pandas as pd
import talib

from features import BaseFeature, Features


class RSIFeature(BaseFeature):
    def __init__(self):
        columns = ["RSI"]
        super().__init__(columns, is_sensitive=True, uses_data=True)

    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        rsi = talib.RSI(df["Close"])[-1]
        return {"RSI": float(rsi)}


Features.add("RSI", RSIFeature())


class MACDFeature(BaseFeature):
    def __init__(self):
        columns = ["MACD"]
        super().__init__(columns, is_sensitive=True, uses_data=True)

    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        macd, signal, hist = talib.MACD(
            df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Check if the last value is not NaN
        if np.isnan(macd[-1]):
            raise RuntimeError("MACD is NaN")
        else:
            return {"MACD": float(macd[-1])}


Features.add("MACD", MACDFeature())


class BollingerBandsFeature(BaseFeature):
    def __init__(self):
        columns = ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
        super().__init__(columns, is_sensitive=True, uses_data=True)

    def _calculate(self, df: pd.DataFrame) -> dict[str, float]:
        window_size = len(df)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df["Close"], timeperiod=window_size
        )
        return {
            "BB_UPPER": float(bb_upper[-1]),
            "BB_MIDDLE": float(bb_middle[-1]),
            "BB_LOWER": float(bb_lower[-1]),
        }


Features.add("BB", BollingerBandsFeature())
