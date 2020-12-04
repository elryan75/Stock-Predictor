import numpy as np
import tensorflow as tf
from tensorflow import keras
import ta
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io
import base64
import datetime


model = keras.models.load_model('Model_Weights')


def preprocess(company, sc):

    # print(company.to_string())
    company = sc.fit_transform(company)
    company = pd.DataFrame(company)
    # df.columns = ['open', 'high', 'low', 'close', 'volume', 'Name']

    company = np.array(company)

    # Separate into X and y, in the target y we only want the closing price
    X = []


    num_features = company.shape[1]

    for i in company:
        features = []
        for j in range(0, num_features):
            features.append(i[j])
        X.append(features)


    X = np.array(X)



    # Now we shift the y array
    X_shifted = []


    # We want to use 14 previous days to predict the next closing price
    # So we use Day1, Day2, Day3..., Days14 to predict Day 15
    n = 14
    for i in range(n, len(X)):
        X_shifted.append(X[i - n: i, : X.shape[1]])


    X_shifted = np.array(X_shifted)


    # Reshape the arrays to use in the LSTM model
    X_shifted = np.reshape(X_shifted, (X_shifted.shape[0], X_shifted.shape[1], X_shifted.shape[2]))


    return X_shifted

def getPrediction(ticker, company):
    company = company.astype(float)
    ta.add_momentum_ta(company, high='High', low='Low', close='Close', volume='Volume', fillna=True)
    columns = ['Volume', 'momentum_stoch', 'momentum_uo', 'momentum_stoch_signal', 'momentum_ao',
               'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_stoch_rsi_d',
               'momentum_stoch_rsi_k', 'momentum_stoch_rsi', 'momentum_kama', 'momentum_tsi', 'momentum_wr']
    company = company.drop(columns, axis=1)
    initial = company['Close'].iloc[-1]
    for i in range(0, 5):
        if(i==1):
            firstPred = company['Close'].iloc[-1].round(4)
        sc = MinMaxScaler()
        processed_data = preprocess(company, sc)
        prediction = model.predict(processed_data)
        prediction = np.c_[prediction, np.zeros((prediction.shape[0], 4))]
        prediction = sc.inverse_transform(prediction)

        print(prediction)
        new_row = {'Close': prediction[0][0], 'High': prediction[0][1], 'Low': prediction[0][2],
                   'Open': prediction[0][3], 'momentum_rsi': prediction[0][4] + 5}
        company = company.append(new_row, ignore_index=True).tail(15)


    # Generate plot
    fig = plt.figure(figsize=(12, 6))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("4 days Forecast of " + str(ticker), fontsize=25)
    axis.set_xlabel("Date", fontsize=15)
    axis.set_ylabel("Price (USD)", fontsize=15)
    axis.grid()
    base = datetime.datetime.today()
    date_list1 = [base - datetime.timedelta(days=x) for x in range(1,11)]
    date_list2 = [base + datetime.timedelta(days=x) for x in range(5)]
    date_list2 = np.append(date_list1[0], date_list2)

    price1 = (company['Close'].values[0:10])[::-1]
    price2 = company['Close'].values[9:15]

    print(price1)
    print(date_list1)

    print(price2)
    print(date_list2)

    '''
    date_list2 = date_list[0:5]
    date_list1 = date_list[5:15]
    date_list1 = np.append(date_list1, date_list2[-1])
    price_list1 = company['Close'].values[0:10][::-1]
    price_list2 = company['Close'].values[10:15]
    price_list1 = np.append(price_list2[-1], price_list1)
    '''
    axis.plot(date_list1, price1, marker='o', color='black', label="Current Price")
    axis.plot(date_list2, price2, marker='o', color='blue', label="Predicted Price")





    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')




    if(initial > firstPred):
        forecast = (((firstPred/initial)-1)*(100)).round(2)
    else:
        forecast = (((firstPred/initial)-1)*(100)).round(2)

    return (forecast, pngImageB64String)




