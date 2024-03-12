import argparse

from Models.RegressionModel import RegressionModel
from stocks import Stock_Data


# Testing Args:  python3 Train.py Models/Saved/test.model Linear -o -s AAPL AMZN


def _get_args():
    """
    Get Args, test call: python3 Train.py Models/Saved/linear1.model Linear -o -s AAPL

    :return: arguments from command line
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Trains models from data collected from stock markets",
        epilog='When entering stocks enter it in form "Market:Stock"',
    )

    parser.add_argument(
        "filename", nargs=1, help="Save the trained model at."
    )  # Save at location

    parser.add_argument(
        "type", nargs=1, help="Choose ML type.", choices=["Linear"]
    )  # Choose ML type

    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrites(if exists) else trains pre-existing model.",
        action="store_true",
    )  # Overwrite

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
        "-t",
        "--split",
        help="Splits training and test data, higher value, more training data. 0-1 value",
        type=float,
        default=0.8,
    )  # Choose amount of data as training data

    parser.add_argument(
        "-p",
        "--period",
        help='Options: [Int+"d","max"], default 5y',
        type=str,
        default="5y",
    )  # Choose amount of data

    args = parser.parse_args()
    return args


def train_linear(args):
    model = None

    if not args.overwrite:
        model = RegressionModel.load_from_file(args.filename[0], if_exists=True)

    if model is None:
        model = RegressionModel()

    test_sets = []
    for stockName in args.stocks:
        print(f"Training: {stockName}")

        stock = Stock_Data(
            stockName, args.period, model.get_features(), normalized=True
        )  # TODO: Add normalized to CLI

        # Split data,
        train_df, test_df = Stock_Data.train_test_split(stock.df, args.split)

        model.train(train_df)
        model.training_stock.append(stockName)
        test_sets.append((stockName, test_df))

    if args.split != 1:
        print("\n\nFinished Training, now testing")
        for stockName, test_df in test_sets:
            print(f"Testing: {stockName}")
            pred_df = model.test_predict(test_df)
            metrics = model.calculate_metrics(pred_df)
            print(str(metrics))
            print("\n[ ")

    model.save_model(args.filename[0])
    print(f"Successfully saved model at: {args.filename[0]}")


def main(args):
    # Linear training
    print(args)
    if args.type[0] == "Linear":
        train_linear(args)


if __name__ == "__main__":
    main(_get_args())
