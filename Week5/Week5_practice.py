#!/usr/bin/python3.5 

# Week5 practice  

# Import packages 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.datasets import mnist 
from keras.models import Sequential 


# Load Data 
( x_train, y_train ), ( x_test, y_test ) = mnist.load_data() 

# Reshape, which has only one channel ( 28, 28, 1 ) 
x_train = x_train.reshape( 60000, 28, 28, 1 ) 
x_test = x_test.reshape( 10000, 28, 28, 1 ) 
x_train_01 = x_train[ y_train <= 1] 
x_test_01 = x_test[y_test <= 1] 

# Output foramtion One-hot encoding 
from keras.utils import np_utils 
y_train_10 = np_utils.to_categorical( y_train, 10 )
y_test_10 = np_utils.to_categorical(y_test, 10 ) 
y_train_01 = y_train[ y_train <= 1] 
y_train_01 = np_utils.to_categorical(y_train_01,2) 
y_test_01 = y_test[y_test <= 1] 
y_test_01 = np_utils.to_categorical(y_test_01,2) 


# ----- Build Model ----- #
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten 
from keras.layers import Conv2D, MaxPooling2D # 2D is no RGB channel 
from keras.optimizers import SGD 


# Origin 0-9 classification 
ConvLayer = [ Conv2D( 32, (3,3), padding = "same", input_shape = (28,28,1) ),
                Activation('relu'), 
                MaxPooling2D( pool_size = (2,2) ), 
                
                Conv2D( 64, (3,3), padding = "same"), 
                Activation( 'relu' ) , 
                MaxPooling2D( pool_size = ( 2,2 ) ) , 
                
                Conv2D( 128, (3,3), padding = "same"), 
                Activation( 'relu' ), 
                MaxPooling2D( pool_size = ( 2,2 ) ), ] 

FCLayer = [ Flatten(), 
            Dense(200), 
            Activation("relu"), 
            
            Dense(10), 
            Activation( 'softmax' ) ] 

model = Sequential( ConvLayer + FCLayer ) 
model.load_weights("../Week3/handwriting_weights_cnn.h5") 
model.summary()  


# Another 0 or 1 classification 
NewFCLayer = [ Flatten(), 
               Dense(100),
               Activation('sigmoid'), 

               Dense(2), 
               Activation( "softmax" ) ]

model_0to1 = Sequential( ConvLayer + NewFCLayer ) 
for layer in ConvLayer: 
    layer.x_trainable = False 

model_0to1.summary() 


# ----- Compile Model ----- #
model_0to1.compile( loss = "mse", optimizer = SGD(lr=0.05), metrics = ["accuracy"] ) 

# ----- Training Model ----- #
model_0to1.fit( x_train_01, y_train_01, epochs = 5, batch_size = 100 ) 

score = model.evaluate( x_test_01, y_test_01 ) 

print( "The loss is %f, the accuracy is %f." %( score[0], score[1] ) ) 
