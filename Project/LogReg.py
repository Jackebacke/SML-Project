import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb

# Import data
from Preprocessing import X, Y, cv, n_fold


K = range(1, 200)
missclassifications = np.zeros(len(K))
for train_index, val_index in cv.split(X):  # Loopar över alla folds
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]

    for j, k in enumerate(K):  # KNN för varje K, lägger till average missclassification
        KNN = skl_nb.KNeighborsClassifier(n_neighbors=k)
        KNN.fit(X_train, Y_train)
        predictions = KNN.predict(X_val)
        missclassifications[j] += np.mean(predictions != Y_val)

missclassifications /= n_fold  # ta average av alla average errors

plt.plot(K, missclassifications)
plt.xlabel("K")
plt.ylabel("Validation error")
plt.title("Validation error for different K")
plt.savefig("Project/figures/KNN_validation_error.png")
plt.show()
