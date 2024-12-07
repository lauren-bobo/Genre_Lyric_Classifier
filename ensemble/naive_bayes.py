from Data.Data_Processes import load_train_data, preprocess, create_splits 

data = load_train_data()

X, Y, vectorizer = preprocess(data)

splits = create_splits(X, Y)

set1 = splits[0]
set2 = splits[1]
set3 = splits[2]
set4 = splits[3]
