#!/usr/bin/python3.5 

# Week6 
# Sequential are used for linearly stack neural models. One connected by one another. 

# Non-linear stack: 1. A combination structure: for a specific layer, the input is given from two or more output of other hidden layers. 
#                   2. The output of the hidden layer is used for other layers' input. 
# -> use identifier "model" 

import numpy as np 
import matplotlib.pyplot as plt 

from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.optimizers import SGD 

from keras.datasets import mnist 
from keras.utils import np_utils 

# Load data 
( x_train, y_train ), ( x_test, y_test ) = mnist.load_data() 
x_train = x_train.reshape( 60000, 784 ) 
x_test = x_test.reshape( 10000, 784 ) 
# One-hot encoding 
y_train = np_utils.to_categorical( y_train, 10 ) 
y_test = np_utils.to_categorical( y_test, 10 ) 


# Model Functional API 
#   Focous only on what is the input and output. 
from keras.models import Model 
from keras.layers import Input 
from keras.layers import concatenate, add 
from keras.layers.core import Lambda 
from keras import backend as K 

def average_func(inputs): 
    return K.mean( inputs, axis=1 ) # fed by tensor 

# Use Lambda to pack function into layer 
avg_layer = Lambda(average_func) 

inputSize = 4 
x = Input( shape=(inputSize,) ) 
y = avg_layer(x) 
avg_model = Model( x, y ) 
avg_model.summary() 

avg_model.compile( loss='mse',optimizer='sgd' ) 
np.array( [ [ 1,2,3,4] ] ).shape # [[1,2,3,4]] means a 4-D vector. 
print(avg_model.predict( np.array( [[1,2,3,4]] ) ) )
print(avg_model.predict( np.array( [[1,2,3,4], [2,3,4,5]] ) ) )


