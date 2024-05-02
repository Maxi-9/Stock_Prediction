from Tools.parse_args import Parse_Args
from Tools.stocks import StockData
from Types.RegressionModel import RegressionModel


@Parse_Args.parser("Predict values using trained ML model.")
@Parse_Args.filename
@Parse_Args.stocks()
def predict(filename, stocks):
    print(filename, stocks)
    # Load model
    model = RegressionModel.load_from_file(filename)

    stock = StockData.stocks_parse(stocks, model.get_features())

    # Predict closing value
    predicted_close = model.predict(stock.df)

    print(
        f"Predicted closing value({stock.df.index[-1]}) for {stocks}: {predicted_close[0]:.2f}"
    )


if __name__ == "__main__":
    predict()
