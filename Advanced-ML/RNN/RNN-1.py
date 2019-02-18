
#Code adapted from CS321 class 
# ## Installation
# 
# 1. If you haven't already installed Python and Jupyter Notebook:   
#     1. Get Python3 from [Python.org](https://www.python.org/downloads/). **Tensorflow does not yet work with Python 3.7, so you _must_ get Python 3.6.** See https://github.com/tensorflow/tensorflow/issues/20517 for updates on 3.7 support.
#     1. In Terminal, run `python3 -m pip install jupyter`
#     1. In Terminal, cd to the folder in which you downloaded this file and run `jupyter notebook`. This should open up a page in your web browser that shows all of the files in the current directory, so that you can open this file. You will need to leave this Terminal window up and running and use a different one for the rest of the instructions.
# 1. If you didn't install keras previously, install it now
#     1. Install the tensorflow machine learning library by typing the following into Terminal:
#     `pip3 install --upgrade tensorflow`
#     1. Install the keras machine learning library by typing the following into Terminal:
#     `pip3 install keras`
# 

# ## Documentation/Sources
# * [Class Notes](https://jennselby.github.io/MachineLearningCourseNotes/#recurrent-neural-networks)
# * [https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/) for information on sequence classification with keras
# * [https://keras.io/](https://keras.io/) Keras API documentation
# * [Keras recurrent tutorial](https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent)

# # Exercise Option 1: Tune an RNN on the IMDB classification problem
# 
# ## The IMDB Dataset
# The [IMDB dataset](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification) consists of movie reviews (x_train) that have been marked as positive or negative (y_train). See the [Word Vectors Tutorial](https://github.com/jennselby/MachineLearningTutorials/blob/master/WordVectors.ipynb) for more details on the IMDB dataset.

from keras.datasets import imdb
from keras.preprocessing import sequence

imdb.load_data() = (imdb_x_train, imdb_y_train), (imdb_x_test, imdb_y_test) 


# For a standard keras model, every input has to be the same length, so we need to set some length after which we will cutoff the rest of the review. (We will also need to pad the shorter reviews with zeros to make them the same length).

cutoff = 500 #In a block of text it cuts off it off at 500 characters 
imdb_x_train_padded = sequence.pad_sequences(imdb_x_train, maxlen=cutoff)
imdb_x_test_padded = sequence.pad_sequences(imdb_x_test, maxlen=cutoff)

imdb_index_offset = 3 # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset


# Classification
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Define our model.
# Unlike last time, when we used convolutional layers, we're going to use an LSTM, a special type of recurrent network.
# Using recurrent networks means that rather than seeing these reviews as one input happening all at one, with the convolutional layers taking into account which words are next to each other, we are going to see them as a sequence of inputs, with one word occurring at each timestep.

imdb_lstm_model = Sequential()
imdb_lstm_model.add(Embedding(input_dim=len(imdb.get_word_index()) + imdb_index_offset,  #TOOD: What does this do
                              output_dim=100, #TODO: What does this do
                              input_length=cutoff))
# return_sequences tells the LSTM to output the full sequence, for use by the next LSTM layer. The final
# LSTM layer should return only the output sequence, for use in the Dense output layer
imdb_lstm_model.add(LSTM(units=32, return_sequences=True)) #32 by 32 units 
imdb_lstm_model.add(LSTM(units=32))
imdb_lstm_model.add(Dense(units=1, activation='sigmoid')) # because at the end, we want one yes/no answer
imdb_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


# Train the model. __This takes 5-15 minutes. You might not want to re-run it unless you are testing out your own changes.__
imdb_lstm_model.fit(imdb_x_train_padded, imdb_y_train, epochs=1, batch_size=64) #TODO: What does batch_size do?


# Assess the model. __This takes 2-10 minutes. You might not want to re-run it unless you are testing out your own changes.__

imdb_lstm_scores = imdb_lstm_model.evaluate(imdb_x_test_padded, imdb_y_test)
print('loss: {} accuracy: {}'.format(*imdb_lstm_scores))

