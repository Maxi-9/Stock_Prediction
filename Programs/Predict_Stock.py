import GetStocks
import Get_args
import LinearRegression

options=Get_args.GetOptions()

if options!="Error":
    StockData=GetStocks.GetStocks(options)

if options=="Error":
    pass
elif (False):
    print(options)
elif (options["-mode"]=="Linear"):
    LinearRegression.LinearPredection(options,StockData)
