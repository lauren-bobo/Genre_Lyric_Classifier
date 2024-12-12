import numpy as np
import pickle
from Data.Data_Processes import load_train_data, preprocess, create_splits
import random
import pandas as pd

data = load_train_data()
X, Y, vectorizer = preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y)
split_1, split_2, split_3, split_4 = splits[:4]

param_combinations = [
    {'max_iter': 2000, 'solver': 'saga', 'penalty': 'elasticnet', 'C': 0.5, 'l1_ratio': 0.5},
    {'max_iter': 3000, 'solver': 'saga', 'penalty': 'elasticnet', 'C': 0.1, 'l1_ratio': 0.7},
    {'max_iter': 2500, 'solver': 'saga', 'penalty': 'elasticnet', 'C': 1.0, 'l1_ratio': 0.3}
]

# Create a pandas DataFrame from the parameter combinations
params_df = pd.DataFrame(param_combinations)
print(params_df)

def train_and_evaluate_model(X_train, Y_train, X_test, Y_test, params):
    print("Training and Evaluating Logistic Regression Model...")
    start_time = time.time()
    
    if 'l1_ratio' in params:
        lr_model = LogisticRegression(
            max_iter=params['max_iter'], 
            solver=params['solver'], 
            penalty=params['penalty'], 
            C=params['C'], 
            n_jobs=-1,
            l1_ratio=params['l1_ratio']
        )
    else:
        lr_model = LogisticRegression(
            max_iter=params['max_iter'], 
            solver=params['solver'], 
            penalty=params['penalty'], 
            C=params['C'], 
            n_jobs=-1
        )
    
    lr_model.fit(X_train, Y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    Y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return lr_model

# test data
X_test, Y_test = split_4

# train, evaluate and save the first model using split_1 with parameters set 1
print("Training and Evaluating Model 1 using split_1 with parameters set 1...")
X_train_1, Y_train_1 = split_1
params_1 = param_combinations[0]
lr_model_1 = train_and_evaluate_model(X_train_1, Y_train_1, X_test, Y_test, params_1)
with open('lr_model_1.pkl', 'wb') as f:
    pickle.dump(lr_model_1, f)

# train, evaluate and save the second model using split_2 with parameters set 2
print("Training and Evaluating Model 2 using split_2 with parameters set 2...")
X_train_2, Y_train_2 = split_2
params_2 = param_combinations[1]
lr_model_2 = train_and_evaluate_model(X_train_2, Y_train_2, X_test, Y_test, params_2)
with open('lr_model_2.pkl', 'wb') as f:
    pickle.dump(lr_model_2, f)

# train, evaluate and save the third model using split_3 with parameters set 3
print("Training and Evaluating Model 3 using split_3 with parameters set 3...")
X_train_3, Y_train_3 = split_3
params_3 = param_combinations[2]
lr_model_3 = train_and_evaluate_model(X_train_3, Y_train_3, X_test, Y_test, params_3)
with open('lr_model_3.pkl', 'wb') as f:
    pickle.dump(lr_model_3, f)