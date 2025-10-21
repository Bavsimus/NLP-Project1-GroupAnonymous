import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load and check data
df = pd.read_csv('Dataset/spam_ham_dataset.csv')
df = df.dropna(subset=['text', 'label'])

X = df['text']
y = df['label']

# TF-IDF Vectorization is a feature that converts text data into numerical format
vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print("=================================")
print("SPAM DETECTION - SVM with TF-IDF")
print("=================================")
print(f"Dataset size: {len(df)} samples")
print(f"Training set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
print(f"Features: {X_tfidf.shape[1]}")

# Grid Search for best parameters
print("\nTraining model with GridSearchCV...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    LinearSVC(random_state=42, max_iter=2000),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

# Test the best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# Results
print("\n" + "=================================")
print("RESULTS")
print("=================================")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Ham: {tn} | False Spam: {fp}")
print(f"  False Ham: {fn} | True Spam: {tp}")

print(f"\nSpam Detection:")
print(f"  Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
print(f"  Recall: {tp / (tp + fn):.4f}")
print(f"  F1-Score: {2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))) if (tp + fp) > 0 else 0:.4f}")

print("\n" + "=================================")
