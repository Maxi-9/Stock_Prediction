from Tools.parse_args import Parse_Args
from Types.RegressionModel import RegressionModel


@Parse_Args.parser("Predict values using trained ML model.")
@Parse_Args.filename
@Parse_Args.seed
@Parse_Args.stocks(multiple=False)
def predict(filename, stocks, seed):
    print(filename, stocks)
    # Load model
    model = RegressionModel.load_from_file(filename)
    model.set_seed(seed)
    df = model.features.get_stocks_parse(stocks)

    # Predict closing value
    pred_date, pred_close = model.predict(df)

    print(f"Predicted closing value({pred_date}) for {stocks}: {pred_close:.2f}")


if __name__ == "__main__":
    predict()
