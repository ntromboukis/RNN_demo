import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# window series to produce input/output pairs
def window_transform_series(series,window_size):
    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    series = scaler.fit_transform(series.reshape(-1,1))
    
    # containers for input/output pairs
    X = []
    y = []
    
    # window data
    count = 0
    for t in range(len(series) - window_size):
        # get input sequence
        temp_in = series[t:t + window_size]
        X.append(temp_in)
        
        # get corresponding target
        temp_target = series[t + window_size]
        y.append(temp_target)
        count+=1
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),)
    
    return X,y

# use RNN to fit and predict time series
def RNN_fit_n_predict(X,y,series):
    # normalize series
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    series = scaler.fit_transform(series.reshape(-1,1))
    
    # split our dataset into training / testing sets
    train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point
    window_size = np.shape(X)[1]
    
    # partition the training set
    X_train = X[:train_test_split,:]
    y_train = y[:train_test_split]

    # keep the last chunk for testing
    X_test = X[train_test_split:,:]
    y_test = y[train_test_split:]
    
    ### form model
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(2)

    # given - build model
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(Dense(8, input_dim=window_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    # run your model!
    model.fit(X_train, y_train, nb_epoch=100, batch_size=2, verbose=0)
    
    ### make prediction
    # generate predictions for training
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # plot original series
    plt.plot(series,color = 'k')

    # plot training set prediction
    split_pt = train_test_split + window_size 
    plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')

    # plot testing set prediction
    plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

    # pretty up graph
    plt.xlabel('day')
    plt.ylabel('(normalized) price of Apple stock')
    plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('off')
    plt.show()