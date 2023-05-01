"""
This is the settings for the project. 

When using the word Int, this means a positive whole number
"""

#Shared Options(All of the programs will use this):

Excel = "Results.xlsx"  # Options: [File Name.xlsx] | Sets the excel spreedsheet name/File path
Stock = "AAPL"          # Options: [Stock Symbol]   | Set your stock.
Period = "max"          # Options: [Int+"d","max"]  | Sets how many days of past data it will use.
Mode = "Linear"         # Options: ["Linear"]       | chooses which data prediction to use.


#Testing options(These settings will purely change the testing program)

Testing_Days = 100                    # Options: [Int]            | Sets how many data points
TestingExcel = "TestingResults.xlsx"  # Options: [File Name.xlsx] | Sets the excel spreedsheet name/File path for testing workbooks
