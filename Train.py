import Settings
from Helper import GetStocks
from Models import RegressionModel

# Linear training
if Settings.TrainMode == "Linear" or Settings.TrainMode == "All":
    model = RegressionModel.loadModel(Settings.ModelPath + "Linear.model")
    if model is None:
        model = RegressionModel.newModel()
    # Gets model
    else:
        model = RegressionModel.newModel()

    for stockName in Settings.TrainStocks:
        print(f"Training: {stockName}")
        StockData = GetStocks.getStocks(stockName, Settings.Period)
        model = RegressionModel.train(StockData, model)

    RegressionModel.saveModel(Settings.ModelPath + "Linear.model", model)
    print(f"Saved model at: {Settings.ModelPath}")
