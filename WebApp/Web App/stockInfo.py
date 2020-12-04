import yfinance as yf
import pandas as pd
import forecast
import ta

def getInfo(ticker):
    tickerData = yf.Ticker(ticker)
    tickerDf = tickerData.history(period='19d').tail(15)
    info = tickerDf[['Close', 'High', 'Low', 'Open', 'Volume']]

    return info


