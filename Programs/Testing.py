#Extra Note, for simplicity: The program will not include the current day
import GetStocks
import LinearRegression
import Settings
import pandas as pd

StockData=GetStocks.GetStocks()
StockDictonary=StockData.to_dict(orient='index')
#(StockDictonary)

with open("Output.txt", "w") as text_file:
            text_file.write(str(StockDictonary))

for i in range(1,Settings.Testing_Days):
    df=StockDictonary.copy()
    StockDictonary.popitem()
    
    LinearRegression.LinearPrediction(pd.DataFrame.from_dict(StockDictonary, orient ='index'),CurrentDate=list(df)[-1],CurVal=df[list(df)[-1]]["Open"],testing=True)
    
    
    
