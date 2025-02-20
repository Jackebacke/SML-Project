import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
# Import data
from Preprocessing import X,Y, random_split 
trainX, trainY, testX, testY = random_split(0.7)

# Define models
naive = DummyClassifier(strategy='most_frequent') # Naive model - always predict the most common class
# Build the best models from method
RandomForest = RandomForestClassifier(n_estimators=100, criterion='entropy') # Random forest



# Compare models
for model in [naive, RandomForest]:
    model.fit(trainX, trainY)
    print(f"Model: {model}")
    print(f"Accuracy: {model.score(testX, testY)}")