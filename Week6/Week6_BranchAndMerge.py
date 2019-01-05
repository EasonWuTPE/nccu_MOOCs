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

# Every layer represents a funcion 
# 1st 
func1 = Dense( 500, activation='sigmoid' ) 
# 2nd 
func2 = Dense( 500, activation='sigmoid' ) 
# 2nd_ use another layer2 
func2_ = Dense( 500, activation='relu' ) 
# 3rd Output layer 
func3 = Dense( 10, activation='softmax' ) 

# Define Input 
x = Input( shape=(784,) ) 
print( "x ", x ) 

# Define the relationship between layers. 
OutputOf1st = func1(x) 
OutputOf2nd = func2( OutputOf1st ) 
OutputOf2nd_ = func2_( OutputOf1st ) 
# Functional API is fed one arg tensor at once. 
#   i.e. y = func3( OutputOf2nd, OutputOf2nd_ ) would ran into error. 
cat_2 = concatenate( [ OutputOf2nd, OutputOf2nd_ ] ) 
y = func3( cat_2 ) 
print( OutputOf1st, OutputOf2nd, OutputOf2nd_, cat_2, y , sep = '\n' ) 

# Build Model 
model = Model( x, y ) # Model API care only on input and output 
model.summary() 

'''
# Model Compile 
model.compile( loss ='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'] ) 
# Model fit 
model.fit( x_train, y_train, batch_size = 100, epochs=5 ) 

# Load Weights to check whether the structure is the same. 
model.load_weights(r'../Week2/handwriting_model_weights.h5' ) 
''' 



