import pickle
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from Data.Data_Processes import load_train_data, preprocess, create_splits

# load and preprocess data
print("Loading and Preprocessing Data...")
data = load_train_data()
X, Y, vectorizer = preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y, num_splits=4)
split_4 = splits[3]  
X_train, Y_train = split_4

import pickle
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from Data.Data_Processes import load_train_data, preprocess, create_splits

# load and preprocess data
print("Loading and Preprocessing Data...")
data = load_train_data()
X, Y, vectorizer = preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y, num_splits=4)
split_4 = splits[3]  
X_train, Y_train = split_4

# load saved models
print("Loading Saved Models...")
model_files = ['dt_model_1.pkl', 'dt_model_2.pkl', 'dt_model_3.pkl',
               'lr_model_1.pkl', 'lr_model_2.pkl', 'lr_model_3.pkl',
               'model_1.pkl', 'model_2.pkl', 'model_3.pkl']
models = [pickle.load(open(file, 'rb')) for file in model_files]
dt_model_1, dt_model_2, dt_model_3, lr_model_1, lr_model_2, lr_model_3, nb_model_1, nb_model_2, nb_model_3 = models

# create the ensemble
print("Creating Stacked Ensemble...")
estimators = [('dt1', dt_model_1), ('dt2', dt_model_2), ('dt3', dt_model_3),
              ('lr1', lr_model_1), ('lr2', lr_model_2), ('lr3', lr_model_3),
              ('nb1', nb_model_1), ('nb2', nb_model_2), ('nb3', nb_model_3)]

# Use Random Forest as the meta-learner
meta_learner = RandomForestClassifier(n_estimators=100, random_state=33)

stacked_ensemble = StackingClassifier(estimators=estimators, final_estimator=meta_learner, n_jobs=-1)

# train the ensemble
print("Training Stacked Ensemble...")
stacked_ensemble.fit(X_train, Y_train)

# evaluate the ensemble
print("Evaluating Stacked Ensemble...")
Y_pred = stacked_ensemble.predict(X_train)
accuracy = accuracy_score(Y_train, Y_pred)
report = classification_report(Y_train, Y_pred)
print(f"Training Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)