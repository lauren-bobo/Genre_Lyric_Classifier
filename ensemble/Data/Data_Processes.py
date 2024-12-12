import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt


# run the first time
nltk.download('stopwords')


# File Paths
Path_To_Test_Data = "C:/Users/ldbob/Downloads/Lyrics_DB/cleaned_test_lyrics.csv"
Path_To_Train_Data =  "C:/Users/ldbob/Downloads/Lyrics_DB/cleaned_train_lyrics.csv"

# Load Data, returns raw train data
def load_train_data():
   print(f"Loading Train Data from: {Path_To_Train_Data}")
   train_data = pd.read_csv(Path_To_Train_Data)
   print(f"Train Data Shape: {train_data.shape}")
   print(f"Train Data Columns: {train_data.columns}")
   return train_data


# Returns raw test data
def load_test_data():
   print(f"Loading Test Data from: {Path_To_Test_Data}")
   test_data = pd.read_csv(Path_To_Test_Data)
   print(f"Test Data Shape: {test_data.shape}")
   return test_data


# Preprocessing Pipeline
def preprocess(data):
   print(f"Preprocessing Data...")
   # Convert lyrics to lowercase
   print(f"Converting to lowercase...")
   data['Lyric'] = data['Lyric'].str.lower()
   # Remove stop words
   data = remove_stop_words(data)
   # Remove punctuation
   data = remove_punctuation(data)
   # Vectorize lyrics with TF-IDF
   X_data, vectorizer = vectorize(data['Lyric'])
   # Encode target labels
   Y_data, encoder = encode_labels(data['genre'])
   print(f"Preprocessing Complete.")
   return X_data, Y_data, vectorizer, encoder


# After preprocessing, Create 4 non-overlapping splits of the data to train each composing model
def create_splits(X, Y, num_splits=4):
    print(f"Creating {num_splits} Splits of the Data...")
    splits = []
    strat_kfold = StratifiedKFold(n_splits=num_splits)
    
    for train_index, _ in strat_kfold.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        splits.append((X_train, Y_train))
    
    return splits


# Preprocessing Helper Functions
def remove_stop_words(data):
   print(f"Removing Stop Words...")
   stop_words = set(stopwords.words('english'))
   # Add pronouns and contractions commonly found in lyrics
   additional_stop_words = { "thats", "theres", "ive", "im", "id", "youre", "verse", "chorus", "vocals", "youve", "shes", "hes", "theyve", "weve", "ill", "youll", "wont", "cant", "isnt", "arent", "wasnt", "werent", "dont", "doesnt", "didnt", "havent", "hasnt", "hadnt", "aint", "gonna", "wanna", "ya", "hey"}
   stop_words.update(additional_stop_words)
   # Remove stop words from lyrics using optimized map
   data['Lyric'] = data['Lyric'].map(
       lambda x: ' '.join([word for word in x.split() if word not in stop_words])
   )
   return data


def remove_punctuation(data):
   print(f"Removing Punctuation...")
    # Remove all punctuation and numeric characters
   data['Lyric'] = data['Lyric'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
   data['Lyric'] = data['Lyric'].str.replace(r'\d+', '', regex=True)      # Remove numbers
   return data


def vectorize(lyrics):
   print(f"Vectorizing Lyrics with TF-IDF...")
   vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Only keep the 5000 most common words
   X_tfidf = vectorizer.fit_transform(lyrics)
   print(f"TF-IDF Shape: {X_tfidf.shape}")
   return X_tfidf, vectorizer


def encode_labels(labels):
   print(f"Encoding Labels...")
   encoder = LabelEncoder()
   y_encoded = encoder.fit_transform(labels)
   print(f"Encoded Labels: {list(encoder.classes_)}")
   return y_encoded


def visualize_tfidf(tfidf_matrix, vectorizer, top_n=100):
   # Convert the TF-IDF matrix to dense and calculate mean scores
   tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
   feature_names = vectorizer.get_feature_names_out()


   # Create a DataFrame of terms and their scores
   tfidf_df = pd.DataFrame({'Term': feature_names, 'Score': tfidf_scores})
   top_terms = tfidf_df.sort_values(by='Score', ascending=False).head(top_n)
  
   # Plot the top terms using Seaborn
   plt.figure(figsize=(10, 6))
   sns.barplot(data=top_terms, x='Score', y='Term', hue='Term', palette='viridis', legend=False)
   plt.title(f'Top {top_n} Terms by Average TF-IDF Score')
   plt.xticks(fontsize=6)
   plt.yticks(fontsize=6)
   plt.xlabel('Average TF-IDF Score')
   plt.ylabel('Terms')
   plt.show()


def plot_top_words_per_genre(data, vectorizer):
   # Create DataFrame from the vectorized features
   X = vectorizer.transform(data['Lyric'])
   words = vectorizer.get_feature_names_out()
   word_freq = pd.DataFrame.sparse.from_spmatrix(X, columns=words)
 
   # Combine with the genre information
   genre_word_freq = pd.concat([data['genre'], word_freq], axis=1)
   genre_word_freq = genre_word_freq.groupby('genre').mean()
  
   # Limit to top 100 words per genre
   top_words = genre_word_freq.iloc[:, :100]

   # Create a heatmap of top words by genre with proper word labels
   plt.figure(figsize=(16, 20))
   sns.heatmap(top_words, cmap='magma', cbar=True, xticklabels=top_words.columns)
   plt.title('Top 100 Words by Genre')
   plt.ylabel('Genre')
   plt.xlabel('Word')
   plt.show()


# Main Execution to test and create diagrams for the data
if __name__ == "__main__":
   # Load Data
   print(f"\nLoading Data...")
   train_data = load_train_data()
   test_data= load_test_data()
  
   trainDF = pd.DataFrame(train_data)
   testDF = pd.DataFrame(test_data)


   sns.countplot(x='genre', data=trainDF)
   plt.show()


   sns.countplot(x='genre', data=testDF)
   plt.show()


   # Check for duplicates
   print(f"Train Data Duplicates: {train_data.duplicated().sum()}")
   print(f"Test Data Duplicates: {test_data.duplicated().sum()}")

   # Preprocess Training Data
   print("\nPreprocessing Training Data...")
   X_train, Y_train, vectorizer = preprocess(train_data)
   print(f"X_train Shape: {X_train.shape}")
   print(f"Y_train Shape: {Y_train.shape}")
  
   # Preprocess Test Data
   print("\nPreprocessing Test Data...")
   X_test = vectorizer.transform(test_data['Lyric'])
   Y_test = encode_labels(test_data['genre'])
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
   visualize_tfidf(X_train, vectorizer, top_n=100)
   print("\nVisualizing Top Words per Genre...")
   plot_top_words_per_genre(train_data, vectorizer)
