from datetime import datetime

import pandas

import Settings
from Helper import GetStocks, ExcelClient, DataChecker
from Models import RegressionModel

# Get Stock Data
df = GetStocks.getStocks(Settings.Stock, Settings.Period)

# Patch for Mondays and Weekends
df = DataChecker.fix(df, Settings.MarketName)
# Debug: print(df[["Open", "Close"]])

# Linear Regression
if Settings.PredictMode == "Linear" or Settings.PredictMode == "All":
    # Get Model
    model = RegressionModel.loadModel(Settings.ModelPath + "Linear.model")

    if model is None:
        # Didn't find Model
        print("Error: Couldn't find Linear Regression Model, Please first train it!")
    else:
        # Start Prediction
        result = RegressionModel.process(df, model)

        # Print Results
        if Settings.PrintResults:
            print(f"Predicted close value for {Settings.Stock} on {result[1]}: {round(result[0], 2)}")
            if result[1] != datetime.today().strftime('%Y-%m-%d'):
                print("Warning: Result may be less accurate, for best results run when market is open.")

        # Excel, it may look complicated, but it is way more simple than it looks
        if Settings.ExcelEnabled:
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

            # Add values`
            ExcelClient.create_row_header(ws, result[1])  # Date
            ExcelClient.write_cell_by_headers(ws, result[1], "Pred. Close", result[0])  # Predicted close

            # Check if it is today
            if datetime.now().strftime('%Y-%m-%d') == result[1]:
                todayData = GetStocks.getTodayStocks(Settings.Stock)

                if not todayData.empty:
                    if result[1] == todayData.index[0].strftime('%Y-%m-%d'):
                        ExcelClient.write_cell_by_headers(ws, result[1], "Open", todayData["Open"][0])

            # While missing values
            missingRow = ExcelClient.get_empty_row_index(ws, "Close")

            while missingRow != -1:
                dateStr = ExcelClient.read_cell(ws, missingRow, 1)
                dateToFind = pandas.to_datetime(dateStr)
                # If it is the last value, it would be accurate
                if dateToFind == df.index[-1]:
                    break

                # Get and output the close value
                try:
                    closeVal = df.at[dateToFind, "Close"]
                    ExcelClient.write_cell(ws, missingRow, ExcelClient.get_column_index(ws, "Close"), closeVal)
                except:
                    break

                deltaProfit = closeVal - ExcelClient.read_cell_by_headers(ws, dateStr, "Open")
                ExcelClient.write_cell_by_headers(ws, dateStr, "Buy?", str(deltaProfit > 0))
                if deltaProfit > 0:
                    ExcelClient.write_cell_by_headers(ws, dateStr, "Profit", deltaProfit)

                # Get the next Missing row
                missingRow = ExcelClient.get_empty_row_index(ws, "Close")

                # Calculate other rows



                

            # Save
            ExcelClient.save_workbook(wb, path)
