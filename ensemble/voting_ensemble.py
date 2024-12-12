from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from Data.Data_Processes import load_train_data, preprocess, create_splits
import pickle
import time

# Load and preprocess data
print("Loading and Preprocessing Data...")
data = load_train_data()
X, Y, vectorizer, encoder = preprocess(data)

# Create data splits
print("Creating Data Splits...")
splits = create_splits(X, Y, num_splits=4)
split_4 = splits[3]  # Use one split for training the stacking model
X_train, Y_train = split_4

# Load all pre-trained models
print("Loading Pre-trained Models...")
model_names = [
    'lr_model_1', 'lr_model_2', 'lr_model_3',
    'dt_model_1', 'dt_model_2', 'dt_model_3',
    'nb_model_1', 'nb_model_2', 'nb_model_3'
]
models = {}
for name in model_names:
    with open(f'ensemble/pickel_jar/{name}.pkl', 'rb') as file:
        models[name] = pickle.load(file)

# Define base estimators for the stacking model
print("Defining Base Models...")
base_estimators = [
    ('lr_1', models['lr_model_1']),
    ('lr_2', models['lr_model_2']),
    ('lr_3', models['lr_model_3']),
    ('dt_1', models['dt_model_1']),
    ('dt_2', models['dt_model_2']),
    ('dt_3', models['dt_model_3']),
    ('nb_1', models['nb_model_1']),
    ('nb_2', models['nb_model_2']),
    ('nb_3', models['nb_model_3'])
]

# Define the stacking model with Random Forest as the final estimator
print("Creating Stacking Model with Random Forest...")
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1),
    cv=5  # Optional: Cross-validation for meta-classifier
)

# Train the stacking model and measure training time
print("Training Stacking Model...")
start_time = time.time()
stacking_model.fit(X_train, Y_train)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Evaluate the stacking model
print("Evaluating Stacking Model...")
Y_pred = stacking_model.predict(X_train)
accuracy = accuracy_score(Y_train, Y_pred)
report = classification_report(Y_train, Y_pred)
print(f"Training Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
