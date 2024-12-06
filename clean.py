import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

nltk.download('stopwords')

# File Paths
Path_To_Test_Data = "C:/Users/ldbob/Downloads/Music_Genre_Classifier/Music_Lyric_DB/DB/cleaned_test_lyrics.csv"
Path_To_Train_data = "C:/Users/ldbob/Downloads/Music_Genre_Classifier/Music_Lyric_DB/DB/cleaned_train_lyrics.csv"

# Load Data
def load_data():
    train_data = pd.read_csv(Path_To_Train_data)
    test_data = pd.read_csv(Path_To_Test_Data)
    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")
    return train_data, test_data

# Preprocessing Pipeline
def preprocess(data):
    # Remove stop words
    data = remove_stop_words(data)
    # Remove punctuation
    data = remove_punctuation(data)
    # Convert lyrics to lowercase
    data['Lyric'] = data['Lyric'].str.lower()
    # Vectorize lyrics
    X_data = vectorize(data['Lyric'])
    # Encode target labels
    Y_data = encode_labels(data['genre'])
    return X_data, Y_data

# Preprocessing Helper Functions
def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    data['Lyric'] = data['Lyric'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    return data

def remove_punctuation(data):
    data['Lyric'] = data['Lyric'].str.replace(r'[^\w\s]', '', regex=True)
    return data

def vectorize(lyrics):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(lyrics)
    print(f"TF-IDF Shape: {X_tfidf.shape}")
    return X_tfidf

def encode_labels(labels):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    print(f"Encoded Labels: {list(encoder.classes_)}")
    return y_encoded

# Main Execution
if __name__ == "__main__":
    # Load Data
    train_data, test_data = load_data()
    
    # Preprocess Training Data
    print("\nPreprocessing Training Data...")
    X_train, Y_train = preprocess(train_data)
    print(f"X_train Shape: {X_train.shape}")
    print(f"Y_train Shape: {Y_train.shape}")
    
    # Preprocess Test Data
    print("\nPreprocessing Test Data...")
    X_test, Y_test = preprocess(test_data)
    print(f"X_test Shape: {X_test.shape}")
    print(f"Y_test Shape: {Y_test.shape}")
