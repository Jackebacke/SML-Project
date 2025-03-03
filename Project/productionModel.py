import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier

# Load data
# Import data
pathTrain = "Project/training_data_vt2025.csv"
train = pd.read_csv(pathTrain, dtype={"ID": str}).dropna().reset_index(drop=True)
Xtrain = train.drop(columns=["increase_stock"])  # Features
Ytrain = train["increase_stock"]  # Output

pathTest = "Project/test_data_spring2025.csv"
Xtest = pd.read_csv(pathTest, dtype={"ID": str}).dropna().reset_index(drop=True)


# Train model
bestTree = tree.DecisionTreeClassifier(criterion='entropy')
finalModel = BaggingClassifier(n_estimators=100, estimator=bestTree, random_state=0)
finalModel.fit(Xtrain, Ytrain)

# Test model
predictions = finalModel.predict(Xtest)

# Save as csv
output = pd.DataFrame([1 if p == 'high_bike_demand' else 0 for p in predictions]).T
output.to_csv("Project/predictions.csv", index=False, header=False)


