#!/usr/bin/python3.5 

# Week3 CNN 

'''
 Convolutional Neuron Network 

	Convolutional Layer
			|
	  Maxing Pooling
			|
	Convolutional Layer
			|
	  Maxing Pooling
			|
		  . . . 

		  	|
	Fully Connected
			|
	One-hot encoding

'''

# Import packages 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.datasets import mnist 
from keras.models import Sequential 


# Load Data 
( x_train, y_train ), ( x_test, y_test ) = mnist.load_data() 

# Load Data 
( x_train, y_train ), ( x_test, y_test ) = mnist.load_data() 

# Reshape, which has only one channel ( 28, 28, 1 ) 
x_train = x_train.reshape( 60000, 28, 28, 1 ) 
x_test = x_test.reshape( 10000, 28, 28, 1 ) 
print( "x_train[9487].shape: ", x_train[9487].shape ) 
print( "x_train.shape: ", x_train.shape ) 
print( "x_train[9487]:\n ", x_train[9487] ) 
print( "x_train[9487][:,:,0]: \n", x_train[9487][:,:,0] ) 
plt.imshow( x_train[9487][:,:,0], cmap="Greys" ) 
plt.show() 

# Outpit foramtion One-hot encoding 
from keras.utils import np_utils 
y_train = np_utils.to_categorical( y_train, 10 )
y_test = np_utils.to_categorical(y_test, 10 ) 


# ----- Build Model ----- #
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten 
from keras.layers import Conv2D, MaxPooling2D # 2D is no RGB channel 
from keras.optimizers import SGD 

model = Sequential() 
# First Convolution Layer
# 32 filters with 3*3 size 
model.add( Conv2D( 32, (3,3), padding = "same", input_shape = (28,28,1) ) )
model.add( Activation( 'relu' ) ) 
model.add( MaxPooling2D( pool_size = ( 2,2 ) ) ) 

# Second Convolution Layer
# 64 filters with 3*3 size , filters are usually more than previous layers 
model.add( Conv2D( 64, (3,3), padding = "same") ) 
model.add( Activation( 'relu' ) ) 
model.add( MaxPooling2D( pool_size = ( 2,2 ) ) )

# Third Convolution Layer
# 128 filters with 3*3 size , filters are usually more than previous layers 
model.add( Conv2D( 128, (3,3), padding = "same") ) 
model.add( Activation( 'relu' ) ) 
model.add( MaxPooling2D( pool_size = ( 2,2 ) ) )

# Flatten 
model.add( Flatten() ) 
# Fully connected Network 
model.add( Dense( 200 ) ) 
model.add( Activation( 'relu' ) ) 

# Output layer 
model.add( Dense(10) ) 
model.add( Activation( 'softmax' ) ) 



# ----- Compile Model ----- #
model.compile( loss = "mse", optimizer = SGD(lr=0.05), metrics = ["accuracy"] ) 

# Summary 
model.summary() 
'''
	Layers I: 
		(3*3+1)*32 = 320 parameters
	Layers II:
		...
'''

# ----- Training Model ----- #
model.fit( x_train, y_train, batch_size = 100, epochs = 12 )  


# ----- Testing Model ----- #
score = model.evaluate( x_test, y_test ) 
print( "loss: ", score[0], 
	   "acc: ", score[1] ) 
	

# Save the result
model_json = model.to_json() 
open( "handwriting_model_cnn.json", 'w' ).write( model_json ) 
model.save_weights("handwriting_weights_cnn.h5") 

'''
# Check the result 
predict = model.predict_classes( x_test ) 

pick = np.random.randint( 1, 9999 , 5 ) 
for i in range(5): 
	plt.subplot( 1, 5, i+1 ) 
	plt.imshow( x_test[pick[i]].reshape(28,28), cmap = "Greys" ) 
	plt.title( predict[pick[i]] ) 
	plt.axis("off") 

plt.show()
'''
