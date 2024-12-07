import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from Data_Processes import load_train_data, preprocess

# Load dataset from "clean.py"
data = load_train_data()

X_train, Y_train, vectorizer = preprocess(data)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models from files
dt1 = joblib.load('path_to_decision_tree_model_1.pkl')
dt2 = joblib.load('path_to_decision_tree_model_2.pkl')
dt3 = joblib.load('path_to_decision_tree_model_3.pkl')
lr1 = joblib.load('path_to_logistic_regression_model_1.pkl')
lr2 = joblib.load('path_to_logistic_regression_model_2.pkl')
lr3 = joblib.load('path_to_logistic_regression_model_3.pkl')
nb1 = joblib.load('path_to_naive_bayes_model_1.pkl')
nb2 = joblib.load('path_to_naive_bayes_model_2.pkl')
nb3 = joblib.load('path_to_naive_bayes_model_3.pkl')

# Create the ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('dt1', dt1), ('dt2', dt2), ('dt3', dt3),
    ('lr1', lr1), ('lr2', lr2), ('lr3', lr3),
    ('nb1', nb1), ('nb2', nb2), ('nb3', nb3)
], voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions
y_pred = ensemble_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Model Accuracy: {accuracy}')