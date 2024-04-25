import argparse

from model import Commons
from stocks import Stock_Data


# Testing Args:  python3 Train.py Models/test.spmodel Linear -o -s AAPL AMZN


def _get_args():
    """
    Get Args, test call: python3 Train.py Types/Models/linear1.model Linear -o -s AAPL

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
        "type", nargs=1, help="Choose ML type.", choices=Commons.model_mapping.keys()
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


def main(args):
    model = None

    if not args.overwrite:
        model = Commons.load_from_file(args.filename[0], if_exists=True)

    if model is None:
        model = Commons.model_mapping[args.type[0]]()

    test_sets = []
    for stockName in args.stocks:
        print(f"Training: {stockName}")

        stock = Stock_Data(stockName, args.period, model.get_features())

        # Split data,
        train_df, test_df = Stock_Data.train_test_split(stock.df, args.split)

        model.train(train_df)

        model.training_stock.append(stockName)
        test_sets.append((stock, test_df.copy()))

    if args.split != 1:
        print("\n\nFinished Training, now testing")
        for stock, test_df in test_sets:
            print(f"Testing: {stock.name}")
            test_df.loc[:, "pred_value"] = model.batch_predict(test_df)
            print(test_df)
            print(model.calculate_metrics(test_df))

    model.save_model(args.filename[0])
    print(f"Successfully saved model at: {args.filename[0]}")


if __name__ == "__main__":
    main(_get_args())
