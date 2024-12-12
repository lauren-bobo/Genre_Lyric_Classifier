from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
import pandas as pd
import numpy as np
import pickle
from Data.Data_Processes import load_train_data, preprocess, create_splits

data = load_train_data()
X, Y, vectorizer = preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y)
split_1, split_2, split_3 = splits[:3]  # Exclude split 4

# function to train and evaluate model using cross-validation
def train_and_evaluate_model_cv(X_train, Y_train):
    print("Training and Evaluating Logistic Regression Model with 10-Fold Cross-Validation...")
    start_time = time.time()
    lr_model = LogisticRegression(max_iter=1000)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr_model, X_train, Y_train, cv=kf, scoring='accuracy')
    print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")
    print(f"Cross-validation Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    lr_model.fit(X_train, Y_train)
    return lr_model

# train, evaluate and save the first model using split_1
print("Training and Evaluating Model 1 using split_1...")
X_train_1, Y_train_1 = split_1
lr_model_1 = train_and_evaluate_model_cv(X_train_1, Y_train_1)
with open('model_1.pkl', 'wb') as f:
    pickle.dump(lr_model_1, f)

# train, evaluate and save the second model using split_2
print("Training and Evaluating Model 2 using split_2...")
X_train_2, Y_train_2 = split_2
lr_model_2 = train_and_evaluate_model_cv(X_train_2, Y_train_2)
with open('model_2.pkl', 'wb') as f:
    pickle.dump(lr_model_2, f)

# train, evaluate and save the third model using split_3
print("Training and Evaluating Model 3 using split_3...")
X_train_3, Y_train_3 = split_3
lr_model_3 = train_and_evaluate_model_cv(X_train_3, Y_train_3)
with open('model_3.pkl', 'wb') as f:
    pickle.dump(lr_model_3, f)