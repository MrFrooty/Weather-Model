from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def get_best_degree(filename):
    df = pd.read_csv(filename)
    
    years = df.iloc[:, 0].values.reshape(-1,1)
    rainfall = df.iloc[:, 1:].values.reshape(-1,1)

    #Choose number of folds
    k=5
    #Create KFold object that shuffles using the k we defined 
    kf = KFold(n_splits=k, shuffle=True)

    #Create a degrees array that stores a value from 1 to 21 to represent the amount of degrees we will be testing
    degrees = []
    for deg in range(21):
        degrees.append(deg)
        
    #Initialize an array that can store all the mean squared error values based on each degree and k-value
    mse_scores = np.zeros((len(degrees), k))

    #Iterate over the degrees of the polynomial
    for i, degree in enumerate(degrees):
        #Create a poly features object with the current degree
        poly = PolynomialFeatures(degree=degree)
        #Iterate over the folds
        for j, (train_index, test_index) in enumerate(kf.split(years)):
            #Split the data into training and test sets for the current fold
            years_train, years_test = years[train_index], years[test_index]
            rainfall_train, rainfall_test = rainfall[train_index], rainfall[test_index]
            
            scaler = StandardScaler()
            year_train_scaled = scaler.fit_transform(years_train)
            year_test_scaled = scaler.transform(years_test)

            #Transform the training and test data using the polynomial features object
            year_train_poly = poly.fit_transform(year_train_scaled)
            year_test_poly = poly.transform(year_test_scaled)

            #Fit & predict the model on the training data
            model = LinearRegression()
            model.fit(year_train_poly, rainfall_train)
            rainfall_pred_poly = model.predict(year_test_poly)
            
            #Invert the transformation on the rainfall_pred_poly values
            rainfall_pred = scaler.inverse_transform(rainfall_pred_poly.reshape(-1, 1)).ravel()
            
            #Calculate the mean squared error for the test data
            mse_scores[i, j] = mean_squared_error(rainfall_test, rainfall_pred)
            
    # Calculate the average mean squared error across all folds for each degree of the polynomial
    mean_mse_scores = np.mean(mse_scores, axis=1)
    # print(mean_mse_scores)

    # Find the index of the degree that provides the best performance
    best_degree_index = np.argmin(mean_mse_scores)

    # Print the best degree and its mean squared error
    print("Best degree: ", degrees[best_degree_index])
    print("Mean squared error: ", mean_mse_scores[best_degree_index])

    return degrees[best_degree_index]