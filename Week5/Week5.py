#!/usr/bin/python3.5 

# Week5 Transfer Learning 

import matplotlib.pyplot as plt 
import numpy as np 

from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.optimizers import SGD 

from keras.utils import np_utils
from keras.datasets import mnist 

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

print( "The total of data x_train  is %d, size is %d x %d." %( x_train.shape ) )
print( "The total of data x_test is %d, size is %d x %d." %( x_test.shape ) )


x_train = x_train.reshape( 60000, 784 ) 
x_test = x_test.reshape( 10000, 784 ) 


x_train_01 = x_train[y_train<=1] 
x_test_01 = x_test[y_test<=1] 

y_train_10 = np_utils.to_categorical( y_train, 10 ) 
y_test_10 = np_utils.to_categorical( y_test, 10 ) 

y_train_01 = y_train[ y_train <= 1 ] 
y_train_01 = np_utils.to_categorical( y_train_01, 2 ) 

y_test_01 = y_test[ y_test<= 1] 
y_test_01 = np_utils.to_categorical(y_test_01, 2 ) 

print( "Total training sample is 60000, %s of which is 0 and 1.\nTotal testing sample is 10000, %s of which is 0 and 1.\nSo the sample's distributions is uniform that." %( x_train_01.shape, x_test_01.shape ) ) 


# Build Model 
'''
# Build Method I: use .add attribute. 
model = Sequential( ) 
# First Hidden layer 
model.add( Dense( 500, input_dim = 784 ) ) 
model.add( Activation("sigmoid") ) 
# Second Hidden layer 
model.add( Dense( 500 ) ) 
model.add( Activation("sigmoid") ) 
# Output layer 
model.add( Dense( 10 ) ) 
model.add( Activation("softmax") ) 


# Build Method II: use list cat.  
FirstLayer = [ Dense( 500, input_dim = 784 ), Activation("sigmoid") ] 
SecondLayer = [ Dense( 500 ) , Activation("sigmoid")  ] 
OutPutLayer = [ Dense( 10 ) , Activation("softmax")  ] 
HiddenLayers = FirstLayer + SecondLayer + OutPutLayer  
model = Sequential(HiddenLayers)
model.summary() 

Method I is more intuition but Method II is more flexible. 
'''

all_except_last = [ Dense( 500, input_dim = 784),
                    Activation("sigmoid"),
                    Dense( 500 ),
                    Activation("sigmoid") ]
output_layer = [ Dense(10), Activation("softmax") ] 

model_0_to_9 = Sequential( all_except_last + output_layer ) 
print( "\n Summary of model_0_to_9: \n " )
model_0_to_9.summary() 

# Read weights 
model_0_to_9.load_weights( "../Week2/handwriting_model_weights.h5" ) 

# Another Model 
new_output_layer = [ Dense(2), Activation("softmax") ] 
model_0_to_1 = Sequential( all_except_last + new_output_layer ) 
print( "\n Summary of model_0_to_1: \n " )
model_0_to_1.summary() 

print( ">> We can find that except the output layer, the name of the other hidden layers are the same.\n" ) 
print( "But the \"Trainable params\" is 644002 and \"Total params\" is 644002. Means that the whole model are trained again. We want to use the same params of the hiiden layers to train the new model.\n " ) 

for layer in all_except_last: 
    layer.trainable = False 

print( "Use the attribute trainable = False to not to re-train the model.\n " ) 
print( "The model summary is: " ) 
model_0_to_1.summary() 
print( "The trainable params reduce from 644002 to 1002 which is params of the output layer.\n" )

# Compile Model 
model_0_to_1.compile( loss='mse', optimizer = SGD(lr=0.01), metrics=["accuracy"] ) 

# Train Model 
model_0_to_1.fit( x_train_01, y_train_01, batch_size = 100, epochs = 5 ) 

# Testing data 
score = model_0_to_1.evaluate( x_test_01, y_test_01 ) 
print( "The Loss is %f, the accuracy is %f."%(score[0], score[1] ) )  




