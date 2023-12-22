# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime

#define 2 empty lists
dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            # Parse the date using datetime and convert to an ordinal number
            date_obj = datetime.strptime(row[0], '%m/%d/%Y')
            dates.append(date_obj.toordinal())

            # Remove the dollar sign and convert to float
            prices.append(float(row[1].replace('$', '')))
        return

def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models
    
    svr_rbf.fit(dates, prices)  # fitting the data points in the models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')  # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')  # plotting the line made by the RBF kernel
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')  # plotting the line made by linear kernel
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')  # plotting the line made by polynomial kernel
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    #plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('aapl.csv')
# Debugging: print the dates and prices to check if they are loaded correctly
#print("Dates:", dates)
#print("Prices:", prices)

# Ensure that the dates list is not empty
#if dates:
#predicted_price = predict_price(dates, prices, dates+ 29)
# Example: Predicting the price for a date 29 days after the first date in your data
if dates:
    future_date = datetime.fromordinal(dates[0] + 29).strftime('%m/%d/%Y')
    print(f"Predicting price for: {future_date}")
    predicted_price = predict_price(dates, prices, [[dates[0] + 29]])
    print(predicted_price)
else:
    print("No data loaded from file.")

print(predicted_price)