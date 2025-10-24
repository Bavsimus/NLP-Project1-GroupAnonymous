import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# 1. DATA LOADING AND SPLITTING
# ----------------------------------------------------
print("Step 1: Loading and Splitting Data...")
# df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/spam_ham_dataset.csv')
df = pd.read_csv('/Dataset/spam_ham_dataset.csv')
df = df.dropna(subset=['text', 'label'])

X = df['text']
y = df['label']

# SPLIT FIRST!
# Using the same 'random_state' and 'stratify' for comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training data: {len(X_train)}, Test data: {len(X_test)}")
print("-" * 40)


# 2. FEATURE ENGINEERING (TASK 1)
# ----------------------------------------------------
print("Step 2: Feature Engineering (Vectorization)")

# Common Parameters
vectorizer_params = {
    'stop_words': 'english',
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95
}

# A. N-grams Representation (CountVectorizer)
print("  a. Training N-grams (CountVectorizer)...")
vectorizer_ngrams = CountVectorizer(**vectorizer_params)
X_train_ngrams = vectorizer_ngrams.fit_transform(X_train)
X_test_ngrams = vectorizer_ngrams.transform(X_test)
print(f"  N-grams feature dimension: {X_train_ngrams.shape[1]}")

# B. TF-IDF Representation
print("  b. Training TF-IDF (TfidfVectorizer)...")
vectorizer_tfidf = TfidfVectorizer(**vectorizer_params)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)
print(f"  TF-IDF feature dimension: {X_train_tfidf.shape[1]}")
print("-" * 40)


# 3. CLASSIFICATION (TASK 2: NAIVE BAYES)
# ----------------------------------------------------
print("Step 3: Classification with Naive Bayes")

# Model A: Naive Bayes + N-grams (Counts)
print("\n--- Model: Naive Bayes + N-grams (Counts) ---")
model_nb_ngrams = MultinomialNB()
model_nb_ngrams.fit(X_train_ngrams, y_train)
y_pred_nb_ngrams = model_nb_ngrams.predict(X_test_ngrams)

print(f"Accuracy: {accuracy_score(y_test, y_pred_nb_ngrams):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_nb_ngrams))


# Model B: Naive Bayes + TF-IDF
print("\n--- Model: Naive Bayes + TF-IDF ---")
model_nb_tfidf = MultinomialNB()
model_nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_nb_tfidf = model_nb_tfidf.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred_nb_tfidf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_nb_tfidf))

print("-" * 40)
print("Results generated for Naive Bayes models.")