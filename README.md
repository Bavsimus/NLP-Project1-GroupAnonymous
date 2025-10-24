# NLP-Project1-GroupAnonymous

## Spam Email Detection Project

This project implements various machine learning techniques for spam email classification.

## Dataset

**Source:** [Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)

- Total: 5,171 emails
- Ham: 3,672 | Spam: 1,499

## Models

### 1. SVM with TF-IDF
- **File:** `SVM_TF_IDF.py`
- **Requirements:** `pandas`, `scikit-learn`
- **Accuracy:** 98.55%
- **F1-Score:** 97.52%
- **Precision:** 96.72%
- **Method:** LinearSVC with GridSearchCV optimization
- **Output:** `Outputs/SVM_TF_IDF_Output.txt`
- **Detailed Explanation:** We used TF-IDF vectorization to convert email text into numerical features, followed by training a Linear Support Vector Classifier (SVC). Hyperparameters were optimized using GridSearchCV to achieve the best performance. It achieved high accuracy and F1-score, indicating effective spam detection capabilities.
- **Note:** *These results were generated with a pipeline that may contain data leakage (vectorizing before splitting). This model is pending re-evaluation with the standardized, leak-free pipeline.*

### 2. Naive Bayes with N-grams and TF-IDF
- **File:** `naive_bayes_plus_tfidf_and_ngrams.ipynb`
- **Method:** Multinomial Naive Bayes.
- **Detailed Explanation:** We implemented Multinomial Naive Bayes, a probabilistic classifier well-suited for text data. We compared its performance using both N-grams (Counts) and TF-IDF feature representations, ensuring a "data leak-free" pipeline by splitting the data before vectorization.
- **Results (N-grams):**
  - **Accuracy:** 93.62%
  - **Spam F1-Score:** 0.90
- **Results (TF-IDF):**
  - **Accuracy:** 93.91%
  - **Spam F1-Score:** 0.90

### 3. Decision Tree with N-grams and TF-IDF
- **File:** `decision_tree_models.ipynb`
- **Method:** Decision Tree Classifier (`max_depth=50`, `random_state=42`).
- **Detailed Explanation:** A Decision Tree model was trained to provide an interpretable baseline. We limited `max_depth` to prevent severe overfitting, which is common for single trees on high-dimensional text data.
- **Results (N-grams):**
  - **Accuracy:** 95.27%
  - **Spam F1-Score:** 0.92
- **Results (TF-IDF):**
  - **Accuracy:** 96.43%
  - **Spam F1-Score:** 0.94

### 4. Random Forest with N-grams and TF-IDF
- **File:** `random_forest_models.ipynb`
- **Method:** Random Forest Classifier (`n_estimators=100`, `random_state=42`).
- **Detailed Explanation:** We used a Random Forest, an ensemble method, to improve upon the single Decision Tree's performance. By building 100 trees, this model reduces variance and typically achieves higher, more robust accuracy.
- **Results (N-grams):**
  - **Accuracy:** 98.36%
  - **Spam F1-Score:** 0.97
- **Results (TF-IDF):**
  - **Accuracy:** 97.87%
  - **Spam F1-Score:** 0.96