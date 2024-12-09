import pickle
from scipy.sparse import vstack
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from Data.Data_Processes import load_train_data, preprocess, create_splits
import time

# load and preprocess data
print("Loading and Preprocessing Data...")
data = load_train_data()  
X, Y, vectorizer = preprocess(data)  

# create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y)  
split_1, split_2, split_3, split_4 = splits

# combine data splits 
print("Combining Data Splits...")
X_train = vstack([split_1[0], split_2[0], split_3[0]]) 
Y_train = np.concatenate([split_1[2], split_2[2], split_3[2]]) 

# create a validation set
print("Creating Validation Set from Training Data...")
from sklearn.model_selection import train_test_split
X_train_part, X_valid, Y_train_part, Y_valid = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)

# train naive bayes model
print("Training Naive Bayes Model...")
start_time = time.time()
nb_model = MultinomialNB()  
nb_model.fit(X_train_part, Y_train_part)  
print(f"Model trained in {time.time() - start_time:.2f} seconds")

# evaluate model
print("Evaluating Naive Bayes Model on Validation Set...")
start_time = time.time()
val_predictions = nb_model.predict(X_valid) 
print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")

# show metrics
val_accuracy = accuracy_score(Y_valid, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("Classification Report on Validation Set:\n", classification_report(Y_valid, val_predictions))

# predictions for split 4 (ensemble)
print("Saving Predictions for Split 4...")
X_split4 = split_4[0]  
Y_split4 = split_4[2]  
split4_predictions = nb_model.predict(X_split4)

with open('nb_predictions_split4.pkl', 'wb') as f:
    pickle.dump(split4_predictions, f)

# save the model with pickle
print("Saving Naive Bayes Model...")
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

# save the TF-IDF vectorizer
print("Saving TF-IDF Vectorizer...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Naive Bayes Model, TF-IDF Vectorizer, and Split 4 Predictions Saved Successfully!")
