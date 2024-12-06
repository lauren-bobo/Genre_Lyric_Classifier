import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset
# Assuming you have a CSV file with 'lyrics' and 'genre' columns
data = pd.read_csv('DB/cleaned_train_lyrics.csv')

# Preprocess data
X = data['lyrics']
y = data['genre']

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
# Create individual classifiers
logistic_regression = LogisticRegression(max_iter=1000)
naive_bayes = MultinomialNB()
decision_tree = DecisionTreeClassifier()

# Create an ensemble of the classifiers
ensemble_classifier = VotingClassifier(
    estimators=[
        ('logistic_regression', logistic_regression),
        ('naive_bayes', naive_bayes),
        ('decision_tree', decision_tree)
    ],
    voting='hard'
)

# Train the ensemble classifier
ensemble_classifier.fit(X_train, y_train)

# Make predictions with the ensemble classifier
y_pred_ensemble = ensemble_classifier.predict(X_test)

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
report_ensemble = classification_report(y_test, y_pred_ensemble)

print(f'Ensemble Accuracy: {accuracy_ensemble}')
print('Ensemble Classification Report:')
print(report_ensemble)