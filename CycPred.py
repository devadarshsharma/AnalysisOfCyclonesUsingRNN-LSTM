# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 21:58:34 2018

@author: Adarsh Sharma Roneel Kumar Vishal Prasad 

This project is mainly looking at how Recurrent Neural Network can be used to predict Sea Surface Temperature into the future,
so that the data obtained can be used to make an educated prediction in cyclone formation.

To make the RNN work, we have used Keras and Tensorflow, which are one of the most widely used RNN library for python programming. 

The dataset was provided by the Fiji Meteorological Center and the dataset can be downloaded from the following link:
http://www.bom.gov.au/oceanography/projects/spslcmp/data/index.shtml 

The code was ontained from the following sites. We actually looked at how the codes worked from various sites and then collaborated into
one. Then we tried to understand the algorithm and make changes to suit our needs.
https://www.kaggle.com/jphoon/bitcoin-time-series-prediction-with-lstm
https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm
http://www.python36.com/predict-cryptocurrency-price-using-lstm/

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing 

#Method to create training and testing dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

#Fix random seed for reproducibility
np.random.seed(7)

iteration = 14
full_text = 'C:\\Users\\User\\MyScript\\WeatherData2013-2017 - Copy.csv'
uptil_winston = 'C:\\Users\\User\\MyScript\\WeatherData2013-2017.csv'

#Importing dataset 
dataset = pd.read_csv(full_text)
dataset = dataset.reindex(index = dataset.index[::-1])
                        
#Creating a flexible x values, since using dates take a lot of time.
x_value = np.arange(1, len(dataset) + 1, 1)

#Taking Average Water Temperature column from the entire dataset. This column actually illustrates the SST.
avg_sst = dataset[['Average Water Temperature']]

#Checking whether there are any NANs in the dataset or not.
dataset['Average Water Temperature'].replace(0,np.nan, inplace = True)
dataset['Average Water Temperature'].fillna(method='ffill', inplace = True)

#Plotting the raw data, Avg Water Temperature.
plt.plot(x_value, avg_sst, 'g', label = 'Average Water Temperature')
plt.legend(loc = 'upper right')
plt.title('Plot showing Average Water Temperature')
plt.xlabel('Frequency')
plt.ylabel('Temperature')
plt.show()

#Reshaping our dataset to be input into the RNN algorithm
values =  dataset[['Average Water Temperature']].values.reshape(-1,1)
values = values.astype('float64')

#LSTMs are sensitive to the scale of the input data. It can be a good practice to rescale the data to the range of 0-to-1, 
#also called normalizing. We can easily normalize the dataset using the MinMaxScaler preprocessing class from the scikit-learn library.
scaler = MinMaxScaler(feature_range = (0,1))
scaled = scaler.fit_transform(values)

#Split test and train
train_size = int(len(scaled) * 0.7)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:] , scaled[train_size:len(scaled),:]
print(len(train),len(test))

#Filling the training and testing dataset with appropriate proportion by call the create_dataset method
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#Reshaping the training data and the test data 
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#This is where the main stuff happens. The LSTM model is created and appropriate parameters are passed into it.
model = Sequential()#Linear stack of layers
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))#shows the number of input and the shape of the array
model.add(Dense(1, activation='sigmoid'))#number of output and the activation function
model.compile(loss='mean_squared_error', optimizer = 'adam')#loss calculation and optimizer is used as adam
history = model.fit(trainX, trainY, epochs=100, batch_size = 100, validation_data = (testX, testY), verbose=0, shuffle=False)

#Plot to show the loss from the training done
plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['val_loss'], label = 'test loss')
plt.legend()
plt.title('Loss vs Val_loss')
plt.show()

#Doing prediction using the test data, and inputing it into the model we trained. 
prediction = model.predict(testX)
plt.plot(prediction, label='Prediction')
plt.plot(testY, label = 'Actual')
plt.title('Plot showing Prediction vs Actual')
plt.xlabel('Frequency')
plt.ylabel('Temperature in Normalised Form')
plt.legend()
plt.show()

#Denormalising
prediction_inverse = scaler.inverse_transform(prediction.reshape(-1,1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1,1))

#Calculating the Root Mean Square Error
rmse = sqrt(mean_squared_error(testY_inverse, prediction_inverse))
print('Test RMSE: %.3f' % rmse) #How similar, on average, are the numbers in list1 to list2

#Plotting after denormalising is done
predictDates = dataset.tail(len(testX)).index
testY_reshape = testY_inverse.reshape(len(testY_inverse))
prediction_reshape = prediction_inverse.reshape(len(prediction_inverse))

plt.plot(predictDates, testY_reshape, label='Actual')
plt.plot(predictDates, prediction_reshape, label='Predict')
plt.legend()
plt.title('Plot showing Prediction vs Actual After Denormalising')
plt.xlabel('Frequency')
plt.ylabel('Temperature')
plt.show()  

#Predicting the future numbers for 7 days.
futurepredictinput = np.array([[prediction[-1]]])
total_of_temp = 0.0
#predict length consecutive values from a real one
print('----------------------------------------------------------------')
for i in range(iteration): 
    futurepredict_in_scaler = model.predict(futurepredictinput)    
    futurepredict = scaler.inverse_transform(futurepredict_in_scaler)
    futurepredictinput = np.array([futurepredict_in_scaler])
    total_of_temp = total_of_temp + futurepredict
    print('Future Prediction for Day ', i + 1, ' is ', futurepredict)


mean_temp = total_of_temp/iteration

print('----------------------------------------------------------------')
if mean_temp > 30:
    print('The mean temperature for the next ' ,iteration, ' day is ', mean_temp)
    print('and unfortunately, by looking at it, we might experience some major tropical depressions')
elif mean_temp > 26.5 and mean_temp <= 30:
    print('The mean temperature for the next ' ,iteration, ' day is ', mean_temp)
    print('The might be some low depressions')
else:
    print('The mean temperature for the next ' ,iteration, ' days is ', mean_temp)
    print('The weather looks cool')

