# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl 

# Update this line to include your API key
#quandl.ApiConfig.api_key = 'YOUR_API_KEY_HERE'


# Retrieving stock data from Quandl for Tata Global Beverages
data = quandl.get("NSE/TATAGLOBAL")## Retrieve data using your API key

# Displaying the top 10 rows of the dataset
data.head(10)

# Plotting the closing price of the stock
plt.figure(figsize=(16,8))
plt.plot(data['Close'], label="Closing Price")

# Calculating price difference between Open and Close, and High and Low
data["Open - Close"] = data["Open"] - data["Close"]
data["High - Low"] = data["High"] - data["Low"]
data = data.dropna()

# Input features for predicting whether to buy or sell the stock
X = data[["Open - Close", "High - Low"]]
X.head()

# Generating labels for stock prediction (1: Buy, -1: Sell)
Y = np.where(data["Close"].shift(-1) > data["Close"], 1, -1)
Y

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

# Importing KNeighborsClassifier and necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Using GridSearch to find the best parameters for KNN classification
params = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn = KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

# Fitting the model
model.fit(X_train, Y_train)

# Calculating accuracy scores for training and testing data
accuracy_train = accuracy_score(Y_train, model.predict(X_train))
accuracy_test = accuracy_score(Y_test, model.predict(X_test))

print("Train_data Accuracy: %.2f" % accuracy_train)
print("Test_data Accuracy: %.2f" % accuracy_test)

# Predicting classes for test data
Predictions_Classification = model.predict(X_test)

# Displaying actual vs predicted classes for test data
actual_predicted_data = pd.DataFrame({"Actual Class": Y_test, "Predicted Class": Predictions_Classification})
actual_predicted_data.head(10)

# Using KNeighborsRegressor for stock price prediction
from sklearn.neighbors import KNeighborsRegressor

# Splitting data for regression
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X, Y, test_size=0.25, random_state=44)

# Using GridSearch to find the best parameters for KNN regression
params = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn_reg = KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)

# Fitting the regression model and making predictions
model_reg.fit(X_train_reg, Y_train_reg)
predictions = model_reg.predict(X_test_reg)

# Calculating Root Mean Square Error (RMSE)
rms = np.sqrt(np.mean(np.power((np.array(Y_test_reg) - np.array(predictions)), 2)))
rms

# Creating a DataFrame to display actual vs predicted close values
valid = pd.DataFrame({"Actual Close": Y_test_reg, "Predicted Close Value": predictions})
valid.head(10)
