
# # Exercise 2c
# Conceptually, the [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) has several _gates_:
# 
# * __Forget gate__: these weights allow some long-term memories to be forgotten.
# * __Input gate__: these weights decide what new information will be added to the context cell.
# * __Output gate__: these weights decide what pieces of the new information and updated context will be passed on to the output.
# 
# It also has a __cell__ that can hold onto information from the current input (as well as things it has remembered from previous inputs), so that it can be used in later outputs.
# 
# Identify which weights in the one_unit_LSTM model are connected with the context and which are associated with the three gates. This is considerably more difficult to do by looking at the inputs and outputs, so you could also treat this as a code reading exercise and look through the keras code to find the answer.
# 
# _Note_: The output from the predict call is what the linked explanation calls $h_{t}$.