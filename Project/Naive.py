import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
# Import data
from Preprocessing import X,Y, random_split 
trainX, trainY, testX, testY = random_split(0.8)

# Define models
naive = DummyClassifier(strategy='most_frequent') # Naive model - always predict the most common class
naive.fit(trainX, trainY)
print("Naive model accuracy: ", naive.score(testX, testY))