import pandas as pd
import numpy as np
from tensorflow import keras
from keras import Sequential, optimizers
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import pylab as p
import seaborn as sns
import math

# Initializing lists for different features
time = []
energy = []
sin = []
cos = []
weekend = []

# List of filenames to load data from
filename = [
    '1518 1000 S', '936 S 400 W St', '601 College Dr', '815 College Dr', '785 W 1000 S', 'College Dr',
    '670 E 1550 N', '240 E 600 S', '349 S 200 E', '357 S 200 E', '475 300 E', '219 Colfax Ave',
    '1407 West North Temple', '2600 Sunnyside Ave S', '1170 E Wilmington Ave', '855 W California Ave',
    '349 W 300 S', '210 E 400 S', '600 E 900 S', '1060 S 900 W', '2375 900 E', '1040 E Sugarmont Dr.',
    '55 East 300 South', '157 Main St', '965 UT-99', '748 E Main St', '6527 N HWY 36', '14814 Minuteman Dr',
    '1874 W 2700 N', '444 Old US Hwy 91', '980 Hoodoo way', '725 E Main St', '248 East 600 South',
    '475 S 300 E', '261 E 500 S', '1990 W 500 S'
]

# Loop through each filename, read the data, and create a prediction model
for i in range(0, len(filename)):
    name = filename[i] + ".csv"
    title = filename[i]
    inagename = filename[i] + "LSTM.png"  # Updated image name to reflect LSTM usage

    df = pd.read_csv(name)
    input_feature = df.iloc[:, 1:8].values
    input_data = input_feature

    lookback = 5
    test_size = 145
    X = []
    y = []

    for i in range(len(df) - lookback - 1):
        t = []
        for j in range(0, lookback):
            t.append(input_data[[(i + j)], :])
        X.append(t)
        y.append(input_data[i + lookback, 6])

    X, y = np.array(X), np.array(y)
    X_test = X[len(X) - test_size:]
    y_test = y[len(X) - test_size:]
    X_train = X[:len(X) - test_size]
    y_train = y[:len(X) - test_size]

    X_train = X_train.reshape(X_train.shape[0], lookback, 7)
    X_test = X_test.reshape(X_test.shape[0], lookback, 7)

    # LSTM Model
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, input_shape=(X_train.shape[1], 7)))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(LSTM(units=30))
    model.add(Dropout(0.5))  # Regularization to avoid overfitting
    model.add(Dense(units=1, activation='relu'))  # Output layer

    model.summary()

    # Compile the model using Stochastic Gradient Descent optimizer
    sgd = optimizers.SGD(lr=0.01, decay=0.000001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=500, batch_size=32)

    # Predict using the test data
    predicted_value = model.predict(X_test)

    # Calculate the maximum power for plotting
    max_power = max(max(predicted_value), max(y_test))

    # Plot the actual vs predicted energy usage
    plt.plot(y_test, predicted_value, 'r.')
    plt.xlabel('Actual Energy used in Kwh')
    plt.ylabel('Predicted Energy used in Kwh')
    plt.title(title)
    p.ylim(0, max_power)
    p.xlim(0, max_power)
    plt.savefig(inagename)
