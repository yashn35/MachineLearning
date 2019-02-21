
#Code adapted from CS321 class 

# # Exercise 2b
# What do each of the six weights of the two_unit_SRNN control? Again, test out your hypotheses carefully.

# ## Exploring LSTMs (Optional Extension to Exercise 2)
one_unit_LSTM = Sequential()
one_unit_LSTM.add(LSTM(units=1, input_shape=(None, 1),
                       activation='linear', recurrent_activation='linear',
                       use_bias=False, unit_forget_bias=False,
                       kernel_initializer='zeros',
                       recurrent_initializer='zeros',
                       return_sequences=True))


one_unit_LSTM_weights = one_unit_LSTM.get_weights()
one_unit_LSTM_weights

one_unit_LSTM_weights[0][0][0] = 1
one_unit_LSTM_weights[0][0][1] = 0
one_unit_LSTM_weights[0][0][2] = 1
one_unit_LSTM_weights[0][0][3] = 1
one_unit_LSTM_weights[1][0][0] = 0
one_unit_LSTM_weights[1][0][1] = 0
one_unit_LSTM_weights[1][0][2] = 0
one_unit_LSTM_weights[1][0][3] = 0
one_unit_LSTM.set_weights(one_unit_LSTM_weights)
one_unit_LSTM.get_weights()

one_unit_LSTM.predict(numpy.array([ [[0], [1], [2], [4]] ]))
