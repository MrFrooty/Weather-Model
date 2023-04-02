import pandas as pd
import numpy as np

#Read file using pandas library
df = pd.read_csv('rainfall in india 1901-2015.csv')

#Clean up data set to only include year number and the 12 months as the columns and the data values from 
#the MADHYA MAHARASHTRA subdivision. This is the 2nd largest state in India
df = df.drop(['index', 'SUBDIVISION','ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'], axis=1)
df = df.loc[2622:2736]

# df["Date"] = pd.to_datetime(df["Year"].astype(str) + df["Month"].astype(str), format="%Y%m")
# df = df.set_index("Date")
# df = df.drop(["Year", "Month"], axis=1)

# df = df.stack().reset_index()
# df.columns = ["Date", "Rainfall"]

# df["Year"] = df["Date"].dt.year
# df["Month"] = df["Date"].dt.month

# df = df.drop("Date", axis=1)

# X = df[["Year", "Month"]]
# y = df["Rainfall"]

# print(X, y)

#Process the data into a new csv file
df.to_csv('rainfall in india processed.csv', index=False)
