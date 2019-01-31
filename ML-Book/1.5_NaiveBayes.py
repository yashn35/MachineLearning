#Naive Bayes Classifier
#Page 71, Introduction to Machine Learning

#BernoullioNB assumes classification or binary data
#MultinomialNB needs count, data with a number
import numpy as np
from matplotlib import pyplot as plt
X = np.array([[0, 1, 0, 1],
            [1, 0 , 1, 1],
            [0, 0, 0, 1],
            [1, 0, 1, 0]])

y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))

"""
Notes about what the code does above:

Page 71
Above array is comparing two different classes -- so for class 0 ([[0, 1, 0, 1],) and 2 ([0, 0, 0, 1])
AND class 1 ([1, 0 , 1, 1]) and 3 ([1, 0, 1, 0]])

As you see it adds up the sum so it counts how many 1's there are in the first 
element of class 0 and 2, as you see there is NO 1 elements in this class, assigning it a sum 0 for array[0] 
It counts the amount of 1's for all of the rest of the classes. Ex. For array[1], it is assigned 2 because there are two 1's in the 1st and 3rd class

It can then use a BernoulliNB to classify and predict probablities with the dataset. 
It might predict based on a class whether something is in the class 0 or class 1 

"""
#What does np.unique function do? Unique elements of an arry

#Plotting on a graph
plt.scatter(X[:, 0], y)
plt.legend(["Class 0", "Class 1"], loc=0)
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()


"""
About the three different types of naive bayes algorithms (GaussianNB, BernoulliNB, MultinomialNB): 
Multinomial is between and ordinal number of classes -- so it can be class 0, 1,2,3,4,5, for example if you had more than two classes
GaussianNB can be applied for any continus data, so 1.2, 5.8, 4.3, etc... This won't ever be used for text classification because text will only be a constant number, it will never be needed for decimal data 
BernoulliNB is used for classification between only two classes or two things -- class 0 and class 1. The above code is for a BernoulliNB.

Naive Bayes is generally used for text analysis. For example spam or not spam -- It associates probabilities to each different class and see which word would have the highest likleyhood for spam using bayes theorom. 
"""

"""
In general, smoothing means if you have a lot of up and down data (looks like a sin wave)
so smoothing would mean to average out the data to make the data more 'clean'
Questions:

What does smoothing mean?
What does alpha in general do for machine learning?
What does having a single parameter mean? 
"""

"""

Dhruv Questions:

Can you explain the difference between recall and presicion?
Why are linear and naive bayes model very fast to predict? What makes them fast predict?
"""