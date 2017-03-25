import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact
from ipywidgets import widgets
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

class MySeries():
    '''
    Time series prediction.  User need only input time series csv
    (note: this should consist of a single row/column without header)
    '''
    def __init__(self):
        # variables for training / predictions
        self.w = 0
        self.series = 0
        self.training_predictions = []
        self.testing_predictions = []
        self.train_test_split = 0
        self.test_periods = 0
        self.window_size = 0
        
    # load data
    def load_data(self,csvname):
        series = np.asarray(pd.read_csv(csvname,header = None))
        
        # normalize series
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        self.series = scaler.fit_transform(series.reshape(-1,1))
    
    # window series to produce input/output pairs
    def window_transform_series(self):
        # containers for input/output pairs
        X = []
        y = []

        # window data
        count = 0
        for t in range(len(self.series) - self.window_size):
            # get input sequence
            temp_in = self.series[t:t + self.window_size]
            X.append(temp_in)

            # get corresponding target
            temp_target = self.series[t + self.window_size]
            y.append(temp_target)
            count+=1

        # reshape each 
        X = np.asarray(X)
        X.shape = (np.shape(X)[0:2])
        y = np.asarray(y)
        y.shape = (len(y),)

        return X,y
   
    # train basic RNN model on time series and make desired predictions
    def train_n_predict(self,test_periods):
        self.test_periods = test_periods
        
        # transform series into input/output pairs
        self.window_size = 14
        X,y = self.window_transform_series()
        
        # split our dataset into training / testing sets
        self.train_test_split = int(np.ceil(2*len(self.series)/float(3)))   # set the split point
        
        # partition the training set
        X_train = X[:self.train_test_split,:]
        y_train = y[:self.train_test_split]

        ### form model and fit to training set ###
        # given - fix random seed - so we can all reproduce the same results on our default time series
        np.random.seed(2)

        # given - build model
        optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # build an RNN to perform regression on our time series input/output data
        model = Sequential()
        model.add(Dense(8, input_dim=self.window_size, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # fit the model!
        model.fit(X_train, y_train, nb_epoch=1000, batch_size=2, verbose=0)

        # store trained fit to training set
        self.training_predictions = model.predict(X_train)

        ### make predictions ###
        self.testing_predictions = []
        y_input = self.series[self.train_test_split:self.train_test_split + self.window_size]
        y_input = list(y_input)
        y_input = [v[0] for v in y_input]
        for t in range(test_periods):            
            y_input = np.asarray(y_input)
            y_input.shape = (1,len(y_input))
            
            # compute and store prediction
            pred = model.predict(y_input)
            self.testing_predictions.append(pred[0])

            # kick out last entry in y_input and insert most recent prediction at front
            y_input = y_input[0,1:]
            y_input = np.append(y_input,pred)
        
        self.testing_predictions = [v[0] for v in self.testing_predictions]   
        
    # plot input series as well as prediction
    def plot_all(self,num_preds):
        # plot original series
        plt.plot(np.arange(len(self.series)),self.series,color = 'k',linewidth = 2)

        # plot prediction on training set
        split_pt = self.train_test_split + self.window_size 
        plt.plot(np.arange(self.window_size,split_pt,1),self.training_predictions,color = 'b')   
    
        # plot prediction 
        preds = self.testing_predictions[:num_preds]
        plt.plot(np.arange(split_pt,split_pt + len(preds),1),preds,color = 'r',linewidth = 2)
      
        # label plot
        plt.xlabel('time period',fontsize = 13)
        plt.ylabel('value',fontsize = 13)
        plt.xticks([])
        plt.yticks([])

    # a general purpose function for running and plotting the result of a user-defined input classification algorithm
    def my_slider(self):   
        def show_fit(num_periods):
            # set parameter value of classifier
            self.plot_all(num_periods) 

        interact(show_fit,num_periods=widgets.IntSlider(min=1,max= self.test_periods,step=0,value=1))