import pandas as pd
import csv
import numpy as np
from tensorflow import keras
from keras import Sequential,optimizers
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
import pylab as p
#from sklearn.preprocessing import MinMaxScaler
import random
import math


time=[]
energy=[]
sin=[]
cos=[]
weekend=[]

filename=['1518 1000 S',
'936 S 400 W St',
'601 College Dr',
'815 College Dr',
'785 W 1000 S',
'College Dr',
'670 E 1550 N',
'240 E 600 S',
'349 S 200 E',
'357 S 200 E',
'475 300 E',
'219 Colfax Ave',
'1407 West North Temple',
'2600 Sunnyside Ave S',
'1170 E Wilmington Ave',
'855 W California Ave',
'349 W 300 S',
'210 E 400 S',
'600 E 900 S',
'1060 S 900 W',
'2375 900 E',
'1040 E Sugarmont Dr.',
'55 East 300 South',
'157 Main St',
'965 UT-99',
'748 E Main St',
'6527 N HWY 36',
'14814 Minuteman Dr',
'1874 W 2700 N',
'444 Old US Hwy 91',
'980 Hoodoo way',
'725 E Main St',
'248 East 600 South',
'475 S 300 E',
'261 E 500 S',
'1990 W 500 S']
for i in range(0,len(filename)):
    name=filename[i]+".csv"
    title=filename[i]
    inagename=filename[i]+"nn.png"
    df=pd.read_csv(name)
#df.sort_values('time',inplace=True)
#print(df.head())
    input_feature= df.iloc[:,1:8].values
    input_data=input_feature
    #print(input_feature.shape)

    # sc= MinMaxScaler(feature_range=(0,1))
    # input_data[:,0:5] = sc.fit_transform(input_feature[:,:])




    test_size = 145
    X = []
    y = []
    for i in range(len(df)  - 1):



        X.append(input_data[ [i], :6])
        y.append(input_data[i , 6])
#
#
#
#
    X, y = np.array(X), np.array(y)
    X_test = X[len(X)-test_size:]
    y_test=y[len(X)-test_size:]
    X_train=X[:len(X)-test_size]
    y_train=y[:len(X)-test_size]
    X = X.reshape(X.shape[0], 6)
    X_train=X_train.reshape(X_train.shape[0], 6)
    X_test = X_test.reshape(X_test.shape[0], 6)
    print(X.shape)
    print(X_test.shape)

    model = Sequential()
    model.add(Dense(units=30, input_dim=6, activation='sigmoid'))
    model.add(Dense(units=30,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1,activation='relu'))
    model.summary()
    #
    # model = Sequential()
    # model.add(LSTM(units=30, return_sequences=True, input_shape=(X_train.shape[1],13)))
    # model.add(LSTM(units=30,return_sequences=True))
    # model.add(LSTM(units=30))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=1,activation='relu'))
    # model.summary()


    sgd=optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10000, batch_size=128)
    predicted_value= model.predict(X_test)
    print(len(predicted_value))
    print(len(y_test))
    max_power=max(max(predicted_value),max(y_test))
    # right=0
    # wrong=0
    # for a in range(0,len(predicted_value)):
    #     predicted_value[a]=math.ceil(predicted_value[a])
    #     if predicted_value[a]==y_test[a]:
    #         right+=1
    #     else:
    #         wrong+=1
    plt.plot(y_test,predicted_value,'r.')
    # print(right,wrong)
    plt.xlabel('Actual Energy used in Kwh')

    plt.ylabel('Predicted Energy used in Kwh')
    plt.title(title)
    p.ylim(0, max_power)
    p.xlim(0,max_power)
    plt.savefig(inagename)




    # plt.plot(predicted_value, color= 'red')
    # plt.plot(y_test, color='green')
    # plt.xlabel('Time in descending order')
    # plt.ylabel('Energy used in Kwh')
    # plt.title(title)
    # plt.savefig(inagename)