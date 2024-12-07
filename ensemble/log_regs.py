import joblib
from sklearn.linear_model import LogisticRegression
from Data.Data_Processes import load_train_data, preprocess, create_splits 

data = load_train_data()

X, Y, vectorizer = preprocess(data)

splits = create_splits(X, Y)

print(splits)