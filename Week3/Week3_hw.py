#!/usr/bin/python3.5 

# hw for Week3 CNN 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Import data 
from keras.datasets import mnist 
( x_train, y_train ), ( x_test, y_test ) = mnist.load_data() 

# Reshape 
x_train = x_train.reshape( 60000, 28, 28, 1 ) 
x_test = x_test.reshape( 10000, 28, 28, 1 ) 
# Output is one-hot encoding 
from keras.utils import np_utils 
y_train = np_utils.to_categorical( y_train, 10 ) 
y_test = np_utils.to_categorical( y_test, 10 ) 


from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D 
from keras.optimizers import SGD 


# Hyperparameters 
filters_1 = 8
activation_ = "relu" 
layer_CNNnum = 3
layer_FCNnum = 2
FCN_Dense = 250
loss_ = "categorical_crossentropy" 
BatchSize = 150
Epoch = 8

#---------- Build Model ---------- #
# First CNN Layers
model = Sequential() 
model.add( Conv2D( filters_1, (4,4), padding = "same", input_shape = (28,28,1) ) ) 
model.add( Activation( activation_ ) ) 
model.add( MaxPooling2D( pool_size = (3,3) ) ) 

for l in range( 1, layer_CNNnum ):
	l *= 2
	model.add( Conv2D( filters_1*l, (4,4), padding = "same" ) ) 
	model.add( Activation( activation_ ) ) 
	model.add( MaxPooling2D( pool_size = (3,3) ) )  

'''
model.add( Conv2D( filters_1*4, (4,4), padding = "same" ) ) 
model.add( Activation( activation_ ) ) 
model.add( MaxPooling2D( pool_size = (3,3) ) )  
'''

# Flatten 
model.add( Flatten() ) 

# FCN 
for layers in range( layer_FCNnum ): 
	model.add( Dense( FCN_Dense, activation = activation_ ) ) 
	model.add( Activation( activation_ ) ) 

# Output Layers 
model.add( Dense(10) ) 
model.add( Activation( 'softmax' ) ) 



#---------- Compile Model ---------- #
model.compile( loss=loss_, optimizer = SGD(lr=0.05), metrics=["accuracy"] ) 

# Model Summary 
model.summary() 

#---------- Training Model ---------- #
model.fit( x_train, y_train, batch_size = BatchSize, epochs = Epoch ) 


#---------- Testing Model ---------- #
score = model.evaluate( x_test, y_test ) 
print( "loss: ", score[0], "acc: ", score[1] ) 



# CHeck the result 
predict = model.predict_classes( x_test ) 

pick = np.random.randint( 1, 9999, 5 ) 
for i in range(5): 
	plt.subplot( 1, 5, i+1 ) 
	plt.imshow( x_test[pick[i]].reshape(28,28), cmap = "Greys" ) 
	plt.title( predict[pick[i]] ) 
	plt.axis("off") 

plt.show()


