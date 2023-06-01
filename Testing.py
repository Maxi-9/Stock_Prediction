from datetime import datetime

import Settings
from Helper import GetStocks, ExcelClient, DataChecker
from Models import RegressionModel

# Get Stock Data
df = GetStocks.getStocks(Settings.Stock, Settings.Period)

# Patch for Mondays and Weekends
df = DataChecker.fix(df, Settings.MarketName)


# Linear Regression
if Settings.TrainMode == "Linear" or Settings.TrainMode == "All":
    # Get Model
    model = RegressionModel.loadModel(Settings.ModelPath + "Linear.model")

    if model is None:
        # Didn't find Model
        print("Error: Couldn't find Linear Regression Model, Please first train it!")
    else:
        results = []

        while results.count < Settings.Testing_Days:
            # Start Prediction
            predResult = RegressionModel.process(df, model)
            # predResult


        path = Settings.PredictExcelDir + "Linear.xlsx"
        wb = ExcelClient.open_workbook(path)  # Work Book
        ws = ExcelClient.open_worksheet(wb, Settings.Stock)  # Work Sheet

        # Create columns if not there
        ExcelClient.create_column_header(ws, "Date")
        ExcelClient.create_column_header(ws, "Open")
        ExcelClient.create_column_header(ws, "Close")
        ExcelClient.create_column_header(ws, "Pred. Close")
        ExcelClient.create_column_header(ws, "Buy?")
        ExcelClient.create_column_header(ws, "Profit")

        for result in results:
            # Add values`
            ExcelClient.create_row_header(ws, result[1])  # Date
            ExcelClient.write_cell_by_headers(ws, result[1], "Pred. Close", result[0])  # Predicted close

            # Check if it is today
            if datetime.now().strftime('%Y-%m-%d') == result[1]:
                todayData = GetStocks.getTodayStocks(Settings.Stock)

                if not todayData.empty:
                    if result[1] == todayData.index[0].strftime('%Y-%m-%d'):
                        ExcelClient.write_cell_by_headers(ws, result[1], "Open", todayData["Open"][0])

        # Save
        ExcelClient.save_workbook(wb, path)

