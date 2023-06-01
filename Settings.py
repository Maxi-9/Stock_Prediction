"""
This is the settings for the project. 

When using the word Int, this means a positive whole number
"""


# Predict.py Options:
PrintResults = True                             # Options: [True, False]     | Should print results of the Prediction
PredictExcelDir = "Results/"                    # Options: [File Path]       | Sets the directory to output Excel files
PredictMode = "All"                             # Options: ["All","Linear"]  | Chooses which model to use.
ForceSameDay = False                            # Options: [True, False]     | Option makes prediction more accurate by making sure it has an opening value


# Testing.py Options(These settings will purely change the testing program):
# Program will use give results to Excel file
Testing_Days = 100                              # Options: [Int]            | Sets how many data points
TestingExcel = "TestResults/"                   # Options: [File Path]      | Sets location of directory to output Excel files
TrainMode = "All"                               # Options: ["All","Linear"] | Chooses which data model to train.


# Train.py options (Options for training sets)
TrainStocks = ["AAPL","AMZN","MSFT","CSCO","DOCU","DLTR","REGN"]       # Options: [List of Stocks]  | Trains using the set
ModelPath = "Models/Saved/"                                            # Options: [File directory]  | Sets the model save directory


# Excel Options (For Testing.py and Predict.py)
ExcelEnabled = True          # Option [True, False] | Should create an Excel files?


# Stock Options
# Note: MarketName is only used for predictions to tell if the stock is open
MarketName = "NASDAQ"                           # Options: [Market Name]            | Market name, make sure you change this accordingly
Stock = "AAPL"                                  # Options: [Stock Symbol]           | Set your stock.
Period = "max"                                  # Options: [Int+"d","max"]          | Sets how many days of past data it will use.




