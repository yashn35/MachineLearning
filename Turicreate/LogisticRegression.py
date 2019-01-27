import pandas as pd 

data = pd.read_csv("/Users/amitnarayan/Coding/Machine-Learning-Portfolio/Turicreate/adult.csv", error_bad_lines=False)
print(data.iloc[0])