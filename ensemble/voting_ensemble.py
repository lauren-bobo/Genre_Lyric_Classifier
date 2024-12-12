import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from Data.Data_Processes import load_train_data, preprocess, create_splits
from sklearn.tree import DecisionTreeClassifier

# load and preprocess data
print("Loading and Preprocessing Data...")
data = load_train_data()
X, Y, vectorizer = preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y, num_splits=4)
split_4 = splits[3]  
X_train, Y_train = split_4

# create the Bagging model
print("Creating Bagging Model...")
base_estimator = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=33)
bagging_model = BaggingClassifier(base_estimator=base_estimator, n_estimators=200, bootstrap=True, random_state=33)

# train the model
print("Training Bagging Model...")
bagging_model.fit(X_train, Y_train)

# evaluate the model
print("Evaluating Bagging Model...")
Y_pred = bagging_model.predict(X_train)
accuracy = accuracy_score(Y_train, Y_pred)
report = classification_report(Y_train, Y_pred)
print(f"Training Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
