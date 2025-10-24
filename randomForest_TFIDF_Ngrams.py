import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

print("--- Step 1: Loading and Splitting Data ---")
# Load the dataset
# df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/NLP_P1/spam_ham_dataset.csv')
df = pd.read_csv('Dataset/spam_ham_dataset.csv')
df = df.dropna(subset=['text', 'label'])

X = df['text']
y = df['label']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training data size: {len(X_train_raw)}")
print(f"Test data size: {len(X_test_raw)}")
print("-" * 50)


print("--- Step 2: Feature Engineering (Vectorization) ---")
vectorizer_params = {
    'stop_words': 'english',
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95
}

# A. N-grams Representation (CountVectorizer)
print("  a. Fitting CountVectorizer (N-grams) on training data...")
vectorizer_ngrams = CountVectorizer(**vectorizer_params)
X_train_ngrams = vectorizer_ngrams.fit_transform(X_train_raw)
X_test_ngrams = vectorizer_ngrams.transform(X_test_raw)
print(f"  N-grams feature dimension: {X_train_ngrams.shape[1]}")

# B. TF-IDF Representation
print("  b. Fitting TfidfVectorizer (TF-IDF) on training data...")
vectorizer_tfidf = TfidfVectorizer(**vectorizer_params)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train_raw)
X_test_tfidf = vectorizer_tfidf.transform(X_test_raw)
print(f"  TF-IDF feature dimension: {X_train_tfidf.shape[1]}")
print("-" * 50)


# --- Step 3: Classification with Random Forest ---
print("--- Step 3: Classification with Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Model 3.A: Random Forest + N-grams (Counts)
print("\n--- Model: Random Forest + N-grams (Counts) ---")
rf_model.fit(X_train_ngrams, y_train)
y_pred_rf_ngrams = rf_model.predict(X_test_ngrams)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_ngrams):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf_ngrams))


# Model 3.B: Random Forest + TF-IDF
print("\n--- Model: Random Forest + TF-IDF ---")
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf_tfidf = rf_model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_tfidf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf_tfidf))
print("-" * 50)
print("Random Forest models completed.")