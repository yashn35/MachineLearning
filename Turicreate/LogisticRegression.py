import pandas as pd 
import sklearn.model_selection.train_test_split
data = pd.read_csv("/Users/amitnarayan/Coding/Machine-Learning-Portfolio/Turicreate/adult.csv", error_bad_lines=False)
print(data)