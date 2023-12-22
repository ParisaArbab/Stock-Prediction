# Stock-Prediction
This code predicts stock prices using Support Vector Regression (SVR), a type of machine learning model suitable for regression problems. It reads historical stock price data from a CSV file and uses this data to train three different SVR models, each with a different kernel:

RBF (Radial Basis Function): This kernel is good for handling non-linear data.
Linear: This kernel is for linear data.
Polynomial: This kernel can model more complex, non-linear relationships.

The code performs the following steps:
1.Data Loading: The get_data function reads stock price data from a CSV file. It extracts dates and prices, with prices adjusted to remove a dollar sign and converted into a floating-point number.
2.Data Preprocessing: The dates are supposed to be converted into a numerical format suitable for regression analysis. However, based on previous messages, this part had issues that needed debugging.
3.Model Training: The predict_price function reshapes the dates into the correct input shape for the SVR models and fits three SVR models to the data.
4.Prediction and Plotting: The models are then used to predict stock prices, and the results are plotted on a graph to visualize the fit of each model to the historical data.
5.Execution and Output: The main part of the script executes these functions, prints the predicted prices for a future date, and generates the plot.

The output of running this script was a set of predicted prices for a future stock price and a plot showing the fit of each model to the historical data.







 
