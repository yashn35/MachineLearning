import turicreate as tc

# Load data
data = tc.SFrame('turicreate.csv')
model = tc.text_classifier.create(data,'Rating',features=['Text'])

print(model.evaluate(data))

#Sentiment Analysis using linear regression
#Adapted from https://apple.github.io/turicreate/docs/userguide/text/analysis.html
