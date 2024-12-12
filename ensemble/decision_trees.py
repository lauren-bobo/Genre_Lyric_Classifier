from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import pandas as pd
import numpy as np
import pickle
import os
from Data.Data_Processes import load_train_data, preprocess, create_splits

data = load_train_data()
X, Y, vectorizer = preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y)
split_1, split_2, split_3, split_4 = splits  # Include split 4

# function to train and evaluate model
def train_and_evaluate_model(X_train, Y_train, X_test, Y_test, params):
    print("Training and Evaluating Decision Tree Model...")
    start_time = time.time()
    dt_model = DecisionTreeClassifier(random_state=42, **params)
    dt_model.fit(X_train, Y_train)
    Y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Training and evaluation completed in {time.time() - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))
    return dt_model

# Ensure the directory exists
os.makedirs('pickel_jar', exist_ok=True)

# Parameter sets
params_set_1 = {'max_depth': 30, 'min_samples_split': 100, 'min_samples_leaf': 50}
params_set_2 = {'max_depth': 25, 'min_samples_split': 150, 'min_samples_leaf': 75}
params_set_3 = {'max_depth': 20, 'min_samples_split': 200, 'min_samples_leaf': 100}

# train, evaluate and save the first model using split_1 and split_4 with params_set_1
print("Training and Evaluating Model 1 using split_1 and split_4 with params_set_1...")
X_train_1, Y_train_1 = split_1
X_test_4, Y_test_4 = split_4
dt_model_1 = train_and_evaluate_model(X_train_1, Y_train_1, X_test_4, Y_test_4, params_set_1)
with open('pickel_jar/dt_model_1.pkl', 'wb') as f:
    pickle.dump(dt_model_1, f)

# train, evaluate and save the second model using split_2 and split_4 with params_set_2
print("Training and Evaluating Model 2 using split_2 and split_4 with params_set_2...")
X_train_2, Y_train_2 = split_2
dt_model_2 = train_and_evaluate_model(X_train_2, Y_train_2, X_test_4, Y_test_4, params_set_2)
with open('pickel_jar/dt_model_2.pkl', 'wb') as f:
    pickle.dump(dt_model_2, f)

# train, evaluate and save the third model using split_3 and split_4 with params_set_3
print("Training and Evaluating Model 3 using split_3 and split_4 with params_set_3...")
X_train_3, Y_train_3 = split_3
dt_model_3 = train_and_evaluate_model(X_train_3, Y_train_3, X_test_4, Y_test_4, params_set_3)
with open('pickel_jar/dt_model_3.pkl', 'wb') as f:
    pickle.dump(dt_model_3, f)