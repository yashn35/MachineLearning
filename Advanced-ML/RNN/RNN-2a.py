# # Exercise 2a
# Figure out what the two weights in the one_unit_SRNN model control. Be sure to test your hypothesis thoroughly. Use different weights and different inputs.

# ## Slightly larger simple recurrent model

import numpy
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import LSTM

two_unit_SRNN = Sequential()
two_unit_SRNN.add(SimpleRNN(units=2, input_shape=(None, 1), activation='linear', use_bias=False)) 
"""input shape(None, 1) specifies that there will be one input node for n number of timesteps 
    which in this case is four below.

"""

#Units is equal to 2 nodes in the nueral network 

two_unit_SRNN_weights = two_unit_SRNN.get_weights()
two_unit_SRNN_weights

two_unit_SRNN_weights[0][0][0] = 0
two_unit_SRNN_weights[0][0][1] = 0

two_unit_SRNN_weights[1][0][0] = 0
two_unit_SRNN_weights[1][0][1] = 0

two_unit_SRNN_weights[1][1][0] = 1
two_unit_SRNN_weights[1][1][1] = 0

two_unit_SRNN.set_weights(two_unit_SRNN_weights)
two_unit_SRNN.get_weights()


# This passes in a single sample with four time steps.
two_unit_SRNN.predict(numpy.array([ [[3], [3], [7], [5]] ]))

"""
I was getting this error:

2019-02-21 14:21:53.347427: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-02-21 14:21:53.350009: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
"""