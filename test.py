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
        stock = Stock_Data(stockName, args.period, model.get_features())
        test_df = stock.df
        print(f"Testing on: {stockName}, len: {len(stock.df)}")

        # Use the model to make predictions
        test_df.loc[:, "pred_value"] = model.batch_predict(stock.df)

        # Calculate metrics
        metrics = model.calculate_metrics(test_df)
        print(f"{metrics}\n\n")


if __name__ == "__main__":
    args = _get_args()
    main(args)
