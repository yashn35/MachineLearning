
# # Exercise Option 3: Set up your own RNN model for the Reuters Classification Problem
# 
# Take the model from exercise 1 (imdb_lstm_model) and modify it to classify the [Reuters data](https://keras.io/datasets/#reuters-newswire-topics-classification).
# 
# Think about what you are trying to predict in this case, and how you will have to change your model to deal with this.

# In[24]:


from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


# In[25]:


(reuters_x_train, reuters_y_train), (reuters_x_test, reuters_y_test) = reuters.load_data()

