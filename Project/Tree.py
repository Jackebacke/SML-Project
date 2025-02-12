import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Import data
from Preprocessing import X, Y, cv, random_split

for i in range(10):
    y1 = np.random.uniform(0, 1, 100)
    y2 = np.random.uniform(0, 1, 100)
    
