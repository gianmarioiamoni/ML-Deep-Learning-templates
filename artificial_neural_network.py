# ARTIFICIAL NEURAL NETWORK

import numpy as np
import pandas as pd
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)

# Encoding categorical data
#
# Label Encoding the "Gender" column (column 2 of features matrix)
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# One Hot Encoding the "Geography" column
#
# (column 1 of features matrix)
# There is not order encoding in Geography data, so we need one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#
# When we use ANN is fundamental to apply features scaling
# to all features of both data set and training set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN
#
# We create as an instance of the class Sequential, which allow to create
# the ANN as a sequence of layers.
# Sequential class is taken from the keras module of the keras library,
# which since TensorFlow 2.0 belongs to TensorFlow.
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
#
# We add layers by using the add() method of the ann
# We use the Dense class, from layers module of keras library
# unit = n. of neurons (based on experimentation, no rule of thumb)
# activation: activation function;
#   - 'relu'=rectify activation function for the hidden layers
#   - 'sigmoid'=sigmoid activation function for the output layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
#
# unit = 1 (we have just 1 output value 0/1)
# activation: Sigmoid for output layers; it gives the probability of 0/1
#             Sigmoid must be used in case of binary prediction
#             SoftMax in case of more than 2 values predictions
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN
#
# We use the compile() method of the ann
# optimizer: optimizer function
#   - 'adam'=adam optimizer
#   - 'sgd'=stochastic gradient descent
# loss: loss function
#   - 'binary_crossentropy'=binary crossentropy
#      when we predict a binary value the loss function must always be binary_crossentropy  
# metrics: metrics function
#   - 'accuracy'=accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
# batch_size: number of observations after which the weights are updated
#             usual valeu = 32
# epochs: number of times that the ANN is trained
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making predictions and evaluating the model
#
# We use the predict() method of the ann
# MAKING THE PREDICTION AND EVALUATE THE MODEL

# Predict if the customer with the following information will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000
#
# The values of the features were all input in a double pair of square brackets.
# That's because the "predict" method always expects a 2D array as the format
# of its inputs.
#
# "France" country was not input as a string in the last column
# but as "1, 0, 0" in the first three columns.
# That's because the predict method expects the one-hot-encoded values
# of the state, and as we see in the first row of the matrix of features X,
# "France" was encoded as "1, 0, 0".
# And be careful to include these values in the first three columns,
# because the dummy variables are always created in the first columns.
#
# We need to apply also the same scaling we used in the training (sc.transform())
# we need only the transform method here (we have not to fit it again)
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Alternative version, to avoid manual transformation
print(ann.predict(sc.transform(ct.transform([[600,"France", le.transform(["Male"]).item() ,40,3,60000,2,1,1,50000]])))>.5)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


