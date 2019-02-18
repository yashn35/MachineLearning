#Code adapted from CS321 class 

#Exercise Option 2: Understand the Weight in RNNs
#Exploring Simple Recurrent Layers 
#Before we dive into something as complicated as LSTMs, Let's take a deeper look at simple recurrent layer weights.

import numpy
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import LSTM

"""
The neurons in the recurrent layer pass their output to the next layer, but also back to themselves. 
    The input shape says that we'll be passing in one-dimensional inputs of unspecified length 
    (the None is what makes it unspecified).
"""
one_unit_SRNN = Sequential()
one_unit_SRNN.add(SimpleRNN(units=1, input_shape=(None, 1), activation='linear', use_bias=False))

one_unit_SRNN_weights = one_unit_SRNN.get_weights()
#one_unit_SRNN_weights


#We can set the weights to whatever we want, to test out what happens with different weight values.
one_unit_SRNN_weights[0][0][0] = 1 #Input Weight
one_unit_SRNN_weights[1][0][0] = 0 #Output Weight 
"""
ANSWER:We know that one_unit_SRNN_weights[0][0][0] is the input weight 
because it return's 7 when mutplying 1*7 for the last input vector. Since it doesn't multiply the output weight
as it is the last timestep it should return 7. 
I also proved that one_unit_SRNN_weights[0][0][0] is the input weight because when I made it equal to 0 and 
one_unit_SRNN_weights[1][0][0] = 1 the output 
"""

one_unit_SRNN.set_weights(one_unit_SRNN_weights)
#one_unit_SRNN.get_weights()


# We can then pass in different input values, to see what the model outputs.
# The code below passes in a single sample that has three time steps.

one_unit_SRNN.predict(numpy.array([ [[3], [3], [7]] ]))

print(one_unit_SRNN.predict(numpy.array([ [[3], [3], [7]] ])))
