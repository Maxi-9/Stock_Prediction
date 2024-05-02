from model import Commons
from parse_args import Parse_Args
from stocks import StockData


# Test Args: python3 test.py Models/test.spmodel -s AMZN


@Parse_Args.parser("Test ML model.")
@Parse_Args.filename
@Parse_Args.stocks()
def main(stocks, filename):
    model = Commons.load_from_file(filename)

    for stockName in stocks:
        stock = StockData.stocks_parse(stockName, model.get_features())

        test_df = stock.df
        print(f"Testing on: {stockName}, len: {len(stock.df)}")

        # Use the model to make predictions
        test_df.loc[:, "pred_value"] = model.batch_predict(stock.df)

        # Calculate metrics
        metrics = model.calculate_metrics(test_df)
        print(f"{metrics}\n\n")


if __name__ == "__main__":
    main()
