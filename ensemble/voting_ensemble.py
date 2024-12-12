import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from Data.Data_Processes import load_train_data, preprocess, create_splits
from sklearn.tree import DecisionTreeClassifier

# load and preprocess data
print("Loading and Preprocessing Data...")
data = load_train_data()
X, Y, vectorizer, encoder= preprocess(data)

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y, num_splits=4)
split_4 = splits[3]  
X_train, Y_train = split_4

# load pre-trained models
print("Loading Pre-trained Models...")
model_names = ['lr_model_1', 'lr_model_2', 'lr_model_3', 'dt_model_1', 'dt_model_2', 'dt_model_3', 'nb_model_1', 'nb_model_2', 'nb_model_3']
models = {}
for name in model_names:
    with open(f'ensemble/pickel_jar/{name}.pkl', 'rb') as file:
        models[name] = pickle.load(file)

lr_model_1 = models['lr_model_1']
lr_model_2 = models['lr_model_2']
lr_model_3 = models['lr_model_3']
dt_model_1 = models['dt_model_1']
dt_model_2 = models['dt_model_2']
dt_model_3 = models['dt_model_3']
nb_model_1 = models['nb_model_1']
nb_model_2 = models['nb_model_2']
nb_model_3 = models['nb_model_3']

# create the Bagging model
print("Creating Bagging Model...")
base_estimators = [lr_model_1, lr_model_2, lr_model_3, dt_model_1, dt_model_2, dt_model_3, nb_model_1, nb_model_2, nb_model_3]
bagging_model = BaggingClassifier(base_estimator=base_estimators, n_estimators=200, bootstrap=True, random_state=33)

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
