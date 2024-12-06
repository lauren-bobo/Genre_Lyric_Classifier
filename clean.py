import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

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
    X_data, vectorizer = vectorize(data['Lyric'])
    # Encode target labels
    Y_data = encode_labels(data['genre'])
    return X_data, Y_data, vectorizer

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
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Only keep the 5000 most common words
    X_tfidf = vectorizer.fit_transform(lyrics)
    print(f"TF-IDF Shape: {X_tfidf.shape}")
    return X_tfidf, vectorizer

def encode_labels(labels):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    print(f"Encoded Labels: {list(encoder.classes_)}")
    return y_encoded

def visualize_tfidf(tfidf_matrix, vectorizer, top_n=100):
    # Convert the TF-IDF matrix to dense and calculate mean scores
    tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names()

    # Create a DataFrame of terms and their scores
    tfidf_df = pd.DataFrame({'Term': feature_names, 'Score': tfidf_scores})
    top_terms = tfidf_df.sort_values(by='Score', ascending=False).head(top_n)

    # Plot the top terms using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_terms, x='Score', y='Term', palette='viridis')
    plt.title(f'Top {top_n} Terms by Average TF-IDF Score')
    plt.xlabel('Average TF-IDF Score')
    plt.ylabel('Terms')
    plt.show()
# Main Execution
if __name__ == "__main__":
    # Load Data
    print(f"\nLoading Data...")
    train_data, test_data = load_data()
    
    trainDF = pd.DataFrame(train_data)
    testDF = pd.DataFrame(test_data)

    sns.countplot(x='genre', data=trainDF)
    plt.show()

    sns.countplot(x='genre', data=testDF)
    plt.show()

    
    # Preprocess Training Data
    print("\nPreprocessing Training Data...")
    X_train, Y_train, vectorizerTrain = preprocess(train_data)
    print(f"X_train Shape: {X_train.shape}")
    print(f"Y_train Shape: {Y_train.shape}")
    
    # Preprocess Test Data
    print("\nPreprocessing Test Data...")
    X_test, Y_test, vectorizerTest = preprocess(test_data)
    print(f"X_test Shape: {X_test.shape}")
    print(f"Y_test Shape: {Y_test.shape}")
    # Plotting the distribution of genres in the training data
    plt.figure(figsize=(10, 6))
    sns.countplot(y=train_data['genre'], order=train_data['genre'].value_counts().index)
    plt.title('Distribution of Genres in Training Data')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.show()

    # Plotting the distribution of genres in the test data
    plt.figure(figsize=(10, 6))
    sns.countplot(y=test_data['genre'], order=test_data['genre'].value_counts().index)
    plt.title('Distribution of Genres in Test Data')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.show()

    # Visualize Top TF-IDF Terms
    print("\nVisualizing Top TF-IDF Terms...")
    visualize_tfidf(X_train, vectorizerTrain, top_n=100)
   
