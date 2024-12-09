import pickle
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import vstack
from Data.Data_Processes import load_train_data, preprocess, create_splits

# load and preprocess data
print("Loading and Preprocessing Data...")
data = load_train_data()
X, Y, vectorizer = preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y, num_splits=4)
split_4 = splits[3]  
X_test = split_4[0]
Y_test = split_4[2]

# load saved models
print("Loading Saved Models...")
with open('decision_tree.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('logreg_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

# create ensemble model
print("Creating Ensemble Model...")
ensemble_model = VotingClassifier(
    estimators=[
        ('decision_tree', dt_model),
        ('logistic_regression', lr_model),
        ('naive_bayes', nb_model)
    ],
    voting='hard'  
)

# train the ensemble model 
print("Combining Data Splits...")
X_train = vstack([splits[0][0], splits[1][0], splits[2][0]])  
Y_train = np.hstack([splits[0][2], splits[1][2], splits[2][2]])  
print("Training Ensemble Model...")
ensemble_model.fit(X_train, Y_train)

# evaluate the ensemble model
print("Evaluating Ensemble Model on Split 4...")
Y_pred = ensemble_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f'Ensemble Model Accuracy: {accuracy:.4f}')

# print classification report
print("Classification Report on Split 4:")
print(classification_report(Y_test, Y_pred))
