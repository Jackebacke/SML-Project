import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Import data
from Preprocessing import X, Y, random_split 


y1 = np.random.randint(0, 10, 100)
y2 = np.random.randint(0, 10, 100)

plt.scatter(y1, y2)
plt.title("Random data")
plt.xlabel("y1")
plt.ylabel("y2")
plt.savefig("Project/Figures/RandomData4.png")
plt.show()

