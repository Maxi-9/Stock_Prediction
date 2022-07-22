import GetStocks
import Get_args
import LinearRegression

options=Get_args.GetOptions()
StockData=GetStocks.GetStocks("AAPL")

if (True):
    print(options)
elif (options["-mode"]=="Linear"):
    LinearRegression.LinearPredection(options,StockData)
elif options=="Error":
    pass