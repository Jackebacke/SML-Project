import pandas as pd
import numpy as np

import sklearn.preprocessing as skl_pre
import sklearn.model_selection as skl_ms

# Import data
path = "training_data_vt2025.csv"
data = pd.read_csv(path, dtype={"ID": str}).dropna().reset_index(drop=True)
X = data.drop(columns=["increase_stock"])  # Features
Y = data["increase_stock"]  # Output

# Make things not random
np.random.seed(0)
n_fold = 10
cv = skl_ms.KFold(
    n_splits=n_fold, random_state=2, shuffle=True
)  # Cross-validation with 10 folds, use by calling cv.split(X)


####################################################################################################
def random_split(percent_train=0.5):
    # Split data into training and test set randomly (50% each by default)
    trainI = np.random.choice(
        data.index, size=int(percent_train * len(data)), replace=False
    )
    trainIndex = data.index.isin(trainI)
    trainX = X.iloc[trainIndex]
    trainY = Y.iloc[trainIndex]
    testX = X.iloc[~trainIndex]
    testY = Y.iloc[~trainIndex]
    return trainX, trainY, testX, testY
