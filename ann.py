# This is an Artificial Neural Network with 2 hidden layers in Keras.
# Author : Akshay Shrimali, Github : @marvin08

from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
from pandas import read_csv

# Seeding is done to make the predictive model re-usable i.e. it generates the same output if it is provided with the same input or dataset.
seed = 9 
np.random.seed(seed)

# Now, we read the contents of the .csv file. (our dataset)
filename = 'BBCN.csv'
dataframe = read_csv(filename)

# Now, we store all the values of our dataset into an array.
array = dataframe.values

# X is the input, so it contains all the values of our dataset.
X = array[:, 0:11] 
# Y is the output, so it contains the target variable.
Y = array[:, 11]

# dataframe.head() -> This function displays the first 5 values of our dataset.

# Now, we create a 4-layer model for our predictive analysis.
# The input and the 2 hidden layers contains the 'rectifier' activation function. ('relu')
# The output layer contains the 'sigmoid' activation function as the output is either 0 or 1.
model = Sequential()
model.add(Dense(11, input_dim=11, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# To compile our model, we use the 'binary_crossentropy' to encounter for the cost or loss.
# And 'metrics' is used to measure the performance of our model and 'accuracy' is the best way to do that.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Now, we train our model.
model.fit(X, Y, nb_epoch=200, batch_size=30)

# The 'evaluate' function tells us the accuracy of the model.
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
