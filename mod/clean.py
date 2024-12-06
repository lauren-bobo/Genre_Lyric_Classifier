import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


Path_To_Test_Data = "C:/Users/ldbob/Downloads/Music_Genre_Classifier/Music_Lyric_DB/DB/cleaned_test_lyrics.csv"
Path_To_Train_data = "C:/Users/ldbob/Downloads/Music_Genre_Classifier/Music_Lyric_DB/DB/cleaned_train_lyrics.csv"

def load_data():
    train_data = pd.read_csv(Path_To_Train_data)
    test_data = pd.read_csv(Path_To_Test_Data)
    return train_data, test_data


load_data()

