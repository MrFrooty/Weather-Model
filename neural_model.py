import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

weather_data = pd.read_csv('rainfall in india processed.csv')

years = weather_data["YEAR"].values
months = [1,2,3,4,5,6,7,8,9,10,11,12]
x = []
for year in years:
    year_array = []
    for month in months:
        year_array.append(f"{month}/{year}")
    x.append(year_array)

weather_data = weather_data.drop(columns=["YEAR"])
y = weather_data.iloc[:, :].values

x = np.array(x)
y = np.array(y)
print(x,y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std

model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=2)

loss, accuracy = model.evaluate(x_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
