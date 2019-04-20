#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Dataset Processing
# #### Carl Shan

# This Jupyter Notebook will share more details about how to process your data. Data processing is like preparing the ingredients before cooking; if you prepare them poorly (e.g., leave things half-peeled and dirty) , the meal will taste poor no matter how skillful a chef you are. 
# 
# It's similarly true in machine learning. Dataset processing can be one of the most important things you can do to get your model to perform well.

# #### Introducing some helpful "magic" Jupyter commands
# ? - this will bring up the documentation of a function

# In[80]:


import pandas as pd
from sklearn import preprocessing

get_ipython().run_line_magic('pylab', 'inline')


# Download the [student performance data](http://archive.ics.uci.edu/ml/machine-learning-databases/00320/) and change the path below to wherever you put the data.

# In[81]:


data = pd.read_csv('/Users/yasnara/Desktop/NBA_player_of_the_week.csv', sep=',')


# In[82]:


data.head(10)


# #### Converting Categorical Values to Numerical Ones
# 
# Looking at the data above, we want to convert a number of the columns from categorical to numerical. Most machine learning models deal with numbers and don't know how to model data that is in text form. As a result we need to learn how to do things such as e.g., convert the values in the `school` column to numbers.

# #### First, let's see what values there are in the `school` column

# In[83]:


# This shows a list of unique values and how many times they appear
data['Conference'].value_counts()


# In[84]:


# Converting values in the school column to text
# We are going to define a function that takes a single value and apply it to all the values
def convert_conference(row):
    if row == 'West':
        return 0
    elif row == 'East':
        return 1
    else:
        return None


# In[85]:


# Here's a slow way of using the above function
get_ipython().run_line_magic('time', '')
converted_data = []

for row in data['Conference']:
    new_value = convert_conference(row)
    converted_data.append(new_value)

data['Conference'] = converted_data


# In[86]:


converted_data = data.dropna()
#print (len(converted_data))
#print(converted_data)
#data['Conference'] = converted_data
converted_data.head(10)


# #### Using sklearn's built-in preprocessing module, we can do the same thing

# In[92]:


"""
Above I have converted the conference from east and west to 1.0 and 0.0. With this I can put this into a classification 
algorithim to classify whether given certain features someone is in the western or eastern conference. 
However, this cannot jut go into any type of algorithim due to how the data is constructed (with the current data) a classification 
algorithim that would work, would be a decision trea. Something like logistic regression won't work due to the features 
not being converted to numbers. 
"""


# In[88]:


print(transformed)


# #### Dealing with Null values

# To show you how to deal with null values, I'm going to make some simulated data of students.

# In[89]:


data.tail(10)


# #### One way to deal with null values is to drop them

# In[ ]:


#To summarize what I did in my code -- All the 'East' and 'West' conferences were converted to 0.0 and 1.0. Since there was some NA data I eliminated it, and saved it into a new data set called new_data_2. 
#Let me know if you have any questions about the code. 

