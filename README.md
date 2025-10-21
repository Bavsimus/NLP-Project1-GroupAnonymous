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
- - **Requirements:** `pandas`, `scikit-learn`
- **Accuracy:** 98.55%
- **F1-Score:** 97.52%
- **Precision:** 96.72%
- **Method:** LinearSVC with GridSearchCV optimization
- **Output:** `Outputs/SVM_TF_IDF_Output.txt`
- **Detailed Explanation** We used TF-IDF vectorization to convert email text into numerical features, followed by training a Linear Support Vector Classifier (SVC). Hyperparameters were optimized using GridSearchCV to achieve the best performance. It achieved high accuracy and F1-score, indicating effective spam detection capabilities.

