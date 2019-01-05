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

sampling_dim = 2 
def sampling( args ): 
    z_mean, z_log_var = args 
    epsilon = K.random_normal( shape=(sampling_dim,),mean=0,stddev=1 ) 
    return z_mean+K.exp(z_log_var/2)*epsilon 

# Use Lambda to pack function into layer 
sampling_layer = Lambda( sampling, output_shape=(sampling_dim,)) 

m = Input( shape=(sampling_dim, ) ) 
s = Input( shape=(sampling_dim, ) ) 
z = sampling_layer( [m,s] ) # If the layer function defined by Lambda can use the [] to be input, but Dense cannot. 

# Build Model 
sample_model = Model( [m,s], z ) 

sample_model.summary() 



test_mean = np.random.randint( 10, size=sampling_dim ).reshape(1,2) 
test_log_var = np.array( [0,0] ) 

print( "Avg: {} and {}. ".format( test_mean[0][0], test_mean[0][1] ) ) 





