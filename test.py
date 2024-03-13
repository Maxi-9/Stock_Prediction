import argparse

from model import Commons
from stocks import Stock_Data


# Test Args: python3 test.py Models/test.spmodel -s AMZN
def _get_args():
    parser = argparse.ArgumentParser(
        prog="test.py",
        description="Tests a trained model on given data",
    )

    parser.add_argument(
        "filename", nargs=1, help="Load the trained model from."
    )  # Load from location

    parser.add_argument(
        "-s",
        "--stocks",
        help="Put stock(s) in form: stock.market or just stock",
        action="extend",
        nargs="+",
        type=str,
        required=True,
    )  # Choose Stocks

    parser.add_argument(
        "-p",
        "--period",
        help='Options: [Int+"d","max"], default 5y',
        type=str,
        default="5y",
    )  # Choose amount of data

    args = parser.parse_args()
    return args


def main(args):
    model = Commons.load_from_file(args.filename[0])

    for stockName in args.stocks:
        stock = Stock_Data(
            stockName, args.period, model.get_features(), normalized=True
        )  # TODO: Add normalized to CLI

        print(f"Testing on: {stockName}, len: {len(stock.df)}")

        # Use the model to make predictions
        pred_df = model.test_predict(stock.df)

        pred_df = stock.inv_normalize(pred_df)
        pred_df = stock.inv_normalize_col(pred_df, "pred_value", "Close")
        # Calculate metrics
        metrics = model.calculate_metrics(pred_df)
        print(f"{metrics}\n\n")


if __name__ == "__main__":
    args = _get_args()
    main(args)
