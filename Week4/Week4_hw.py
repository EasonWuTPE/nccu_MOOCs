#!/usr/bin/python3.5 

# Week4 

# Import packages 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.datasets import imdb 

# fetch data 
( x_train, y_train ), ( x_test, y_test ) = imdb.load_data( num_words = 10000 ) 
print( len(x_train), len(y_train) ) 

'''
print( "x_train[99]: ", x_train[99], "\ny_train[99]: ", y_train[99] ) 
	Words are convert into numbers.  
	The number represent the frequency of the use of word.
	The smaller the number is, the more frequent the word is.

''' 

''' every comments are not the same length. 
for i in range(10): 
	print( len(x_train[i]), end = ',' ) 
'''

# deal the data 
from keras.preprocessing import sequence 
# restrict the length of the comments under 100. 
x_train = sequence.pad_sequences(x_train, maxlen = 100 ) 
x_test = sequence.pad_sequences(x_test, maxlen = 100 ) 

from keras.models import Sequential 
from keras.layers import Dense, Embedding 
from keras.layers import LSTM 
# Build Model 
model = Sequential() 
# Word Embedding from 10k-d to 128-d vector 
model.add( Embedding(10000, 128) ) 
# 150 LSTM cells 
model.add( LSTM(150) ) 
model.add( Dense( 1, activation = "sigmoid" ) ) 

# Summary Model 
model.summary()
'''
	3 gates, every gates has 128, 150 cells input, 150 cells
	3*(128+150+1)*150 = 125550
	(128+150+1)*150 = 41850
		=> 125550 + 41850 = 167400 
'''
# Compile model 
model.compile( loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"] ) 

# Train Model 
model.fit( x_train, y_train, batch_size = 32, epochs = 15 ) 

# Evaluate 
score = model.evaluate( x_test, y_test ) 

print( "Accuracy: ", score[1], "\n", "Loss: ", score[0] ) 

'''
# Saveing the results 
model_json = model.to_json() 
open( "imdb_mode_rnn_LSTM_architecture.json", 'w' ).write(model_json) 
model.save_weights( 'imdb_model_weights.h5' ) 
'''

# Another Saving methods 
model.save("my_imdb_lstm.h5") 



