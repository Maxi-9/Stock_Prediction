from Tools.parse_args import Parse_Args
from model import Commons


# Test Args: python3 test.py Models/test.spmodel -s AMZN


@Parse_Args.parser("Test ML model.")
@Parse_Args.filename
@Parse_Args.seed
@Parse_Args.stocks()
def main(stocks, filename, seed):
    model = Commons.load_from_file(filename)

    for stockName in stocks:
        print(f"Testing: {stockName}")
        df = model.features.get_stocks_parse(stockName)

        model.set_seed(seed)
        print(f"Testing on: {stockName}, len: {len(df)}")

        # Use the model to make predictions
        df.loc[:, "pred_value"] = model.batch_predict(df)

        # Calculate metrics
        metrics = model.calculate_metrics(df)
        print(f"{metrics}\n\n")


if __name__ == "__main__":
    main()
