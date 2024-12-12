from sklearn.naive_bayes import MultinomialNB
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
split_1, split_2, split_3, split_4 = splits[:4]

# function to train and evaluate model
def train_and_evaluate_model(X_train, Y_train, X_test, Y_test):
    print("Training Naive Bayes Model...")
    start_time = time.time()
    nb_model = MultinomialNB()
    nb_model.fit(X_train, Y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    print("Evaluating Model...")
    Y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return nb_model

# test data
X_test, Y_test = split_4

# train, evaluate and save the first model using split_1
print("Training and Evaluating Model 1 using split_1...")
X_train_1, Y_train_1 = split_1
nb_model_1 = train_and_evaluate_model(X_train_1, Y_train_1, X_test, Y_test)
with open('model_1.pkl', 'wb') as f:
    pickle.dump(nb_model_1, f)

# train, evaluate and save the second model using split_2
print("Training and Evaluating Model 2 using split_2...")
X_train_2, Y_train_2 = split_2
nb_model_2 = train_and_evaluate_model(X_train_2, Y_train_2, X_test, Y_test)
with open('model_2.pkl', 'wb') as f:
    pickle.dump(nb_model_2, f)

# train, evaluate and save the third model using split_3
print("Training and Evaluating Model 3 using split_3...")
X_train_3, Y_train_3 = split_3
nb_model_3 = train_and_evaluate_model(X_train_3, Y_train_3, X_test, Y_test)
with open('model_3.pkl', 'wb') as f:
    pickle.dump(nb_model_3, f)