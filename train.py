from Tools.parse_args import Parse_Args
from Tools.stocks import StockData
from model import Commons


# Testing Args:  python3 Train.py Models/test.spm Linear -o -s AAPL -s AMZN
# Train on AMZN, MSFT, GOOGL, TSLA, META, NVDA, PYPL, ADBE, NFLX, COST, AVGO, TXN, ASML, QCOM, CMCSA, CSCO, INTC, AMAT, LRCX, KLAC, MRVL, ORCL, SNPS, MKSI, AMD, LOGI, CRWD, ZS, DDOG, PANW, PAYC, CHTR, VRSN, SPLK, TEAM, OKTA, WDAY, CRM, TWLO, DOCU, CRWD, ZM, SNOW, COUP, ESTC, CDNS, SWKS, MCHP, DBX, FTNT, NET


@Parse_Args.parser("Train ML model.")
@Parse_Args.filename
@Parse_Args.split
@Parse_Args.overwrite
@Parse_Args.modeltype
@Parse_Args.stocks()
def main(filename, split, overwrite, mtype, stocks):
    model = None

    if not overwrite:
        model = Commons.load_from_file(filename, if_exists=True)

    if model is None:
        model = Commons.model_mapping[mtype]()

    test_sets = []
    for stockName in stocks:
        print(f"Training: {stockName}")

        df = StockData.stocks_parse(stockName, model.get_features())

        # Split data,
        train_df, test_df = StockData.train_test_split(df, split)

        model.train(train_df)

        model.training_stock.append(stockName)
        test_sets.append((stockName, test_df.copy()))

    if split != 1:
        print("\n\nFinished Training, now testing")
        for stock, test_df in test_sets:
            print(f"Testing: {stock}")
            test_df.loc[:, "pred_value"] = model.batch_predict(test_df)
            print(test_df)
            print(model.calculate_metrics(test_df))

    model.save_model(filename)
    print(f"Successfully saved model at: {filename}")


if __name__ == "__main__":
    main()
