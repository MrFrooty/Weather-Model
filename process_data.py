import pandas as pd
import numpy as np

#Read file using pandas library
df = pd.read_csv('SF_Rainfall.csv')

#Clean up data set
df = df.drop(['Season', '-', 'SecondYear'], axis=1)
df = df[['FirstYear', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
df = df.drop([0, len(df)-1])

#Process the data into a new csv file
df.to_csv('modified_data.csv', index=False)
