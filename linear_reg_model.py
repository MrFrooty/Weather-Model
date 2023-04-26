import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Load dataset into file
df = pd.read_csv('modified_data.csv')

#Create correct datavalues
years = df.iloc[:, 0].values.reshape(-1,1)
rainfall = df.iloc[:, 1:].values

# Calculate the train size as 80% of the len of df and then finding that value as the nearest multiple of 12 (seasonal pattern)
train_size = int(len(df) * 0.8) // 12 * 12

# Split the data into training and test sets
years_train, rainfall_train = years[:train_size], rainfall[:train_size]
years_test, rainfall_test = years[train_size:], rainfall[train_size:]

models = {}

#Train an individual model for each month using ever year's dataset
for month in range(12):
    lin_reg = LinearRegression()
    lin_reg.fit(years_train, rainfall_train[:, month])
    models[month] = lin_reg

#Predict the values for each month and output into an array
rainfall_preds = np.zeros_like(rainfall_test)
for month in range(12):
    print(years_test)
    model = models[month]
    rainfall_preds[:, month] = model.predict(years_test)

#First calculate the mse based on predicted and test values and output to an array
#Loop through that array while keeping track of the index and print out error in correspondence to the correct month
mse = metrics.mean_squared_error(rainfall_test, rainfall_preds, multioutput='raw_values')
for month, error in enumerate(mse):
    print(f"MSE for Month {month+1}: {error:.2f}")
   
#Create a subplot that plots all the original and predicted values onto a plot for every month
#with 4 plots in a total of 3 rows over an area of 12x9 inches.  
fig, axs = plt.subplots(ncols = 4, nrows = 3, figsize=(12,9))
fig.suptitle('Error difference between predicted and test values')
for month in range(12):
    row = month//4
    col = month%4
    ax = axs[row][col]
    ax.plot(years_test, rainfall_test[:, month], label= 'Actual', color='blue')
    ax.plot(years_test, rainfall_preds[:, month], label = 'Predicted', color='red')
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (inches)")
    ax.set_title(f"Month {month + 1}")
    
fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
plt.show()