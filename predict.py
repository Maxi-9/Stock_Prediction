import argparse

from Models.RegressionModel import RegressionModel
from stocks import Stock_Data


def get_args():
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Predicts a stock market's value using trained ML model.",
    )

    parser.add_argument(
        "filename", nargs=1, help="Load the trained model from."
    )  # Load from location

    parser.add_argument(
        "-s",
        "--stocks",
        help="Put stock(s) in form: stock.market or just stock",
        # action="extend",
        nargs="+",
        type=str,
        required=True,
    )  # Choose Stock

    parser.add_argument(
        "-p",
        "--period",
        help='Options: [Int+"d","max"], default 5 years',
        type=str,
        default="5y",
    )  # Choose amount of data

    args = parser.parse_args()
    return args


def predict(filename, stockName, period):
    print(filename, stockName, period)
    # Load model
    model = RegressionModel.load_from_file(filename)

    stock = Stock_Data(
        stockName, period, model.get_features(), normalized=True
    )  # TODO: Add normalized to CLI
    print(stock.df)
    # Get stock data (just need latest row)

    # Predict closing value
    predicted_close = model.predict(stock.df)

    predicted_close = stock.inv_normalize_value(
        predicted_close, model.predictOn.columns()[0]
    )

    print(
        f"Predicted closing value({stock.df.index[-1]}) for {stockName}: {predicted_close[0]:.2f}"
    )


if __name__ == "__main__":
    args = get_args()
    predict(args.filename[0], args.stocks[0], args.period)
