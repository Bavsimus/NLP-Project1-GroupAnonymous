import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

print("=== Step 1: Loading and Splitting Data ===")
df = pd.read_csv('/content/spam_ham_dataset.csv')
df = df.dropna(subset=['text', 'label'])

X = df['text']
y = df['label']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training samples: {len(X_train_raw)} | Test samples: {len(X_test_raw)}")
print("-" * 50)


print("=== Step 2: Feature Engineering (TF-IDF & N-grams) ===")
vectorizer_params = {
    'stop_words': 'english',
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95
}
# Count Vectorizer (N-grams)
print("Fitting CountVectorizer (N-grams)...")
vectorizer_ngrams = CountVectorizer(**vectorizer_params)
X_train_ngrams = vectorizer_ngrams.fit_transform(X_train_raw)
X_test_ngrams = vectorizer_ngrams.transform(X_test_raw)
print(f"N-grams feature dimension: {X_train_ngrams.shape[1]}")

# TF-IDF
print("Fitting TfidfVectorizer (TF-IDF)...")
vectorizer_tfidf = TfidfVectorizer(**vectorizer_params)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train_raw)
X_test_tfidf = vectorizer_tfidf.transform(X_test_raw)
print(f"TF-IDF feature dimension: {X_train_tfidf.shape[1]}")
print("-" * 50)
print("=== Step 3: Classification with K-Nearest Neighbors ===")
# K değerini optimize edebilirsin; burada temel 5 alınmıştır.
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)

# A. KNN + N-grams (Counts)
print("\n--- Model: KNN + N-grams (Counts) ---")
knn_model.fit(X_train_ngrams, y_train)
y_pred_ngrams = knn_model.predict(X_test_ngrams)

print(f"Accuracy: {accuracy_score(y_test, y_pred_ngrams):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_ngrams))
# B. KNN + TF-IDF
print("\n--- Model: KNN + TF-IDF ---")
knn_model.fit(X_train_tfidf, y_train)
y_pred_tfidf = knn_model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_tfidf))

print("-" * 50)
print("KNN models completed.")
