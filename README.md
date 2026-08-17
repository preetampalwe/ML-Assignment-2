# Spambase Binary Classification — Machine Learning Assignment 2

## 1. Problem Statement

The objective of this assignment is to implement and compare multiple supervised machine learning classification algorithms for identifying whether an email is **Spam** or **Non-Spam**.

The problem is formulated as a **binary classification problem**, where:

- `0` = Non-Spam
- `1` = Spam

Five classification algorithms were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble Model)

Each model was evaluated using six performance metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

In addition to the machine learning notebook, an interactive Streamlit web application was developed and deployed on Streamlit Community Cloud. The application allows a user to upload test data, select any of the five trained models, and view the corresponding evaluation results.

---

# 2. Dataset Description

## Dataset

The **Spambase** dataset was selected for this assignment from a public dataset repository.

The dataset contains numerical features derived from email messages and is designed to classify emails as Spam or Non-Spam.

The features include:

- Word frequency features
- Character frequency features
- Capital letter statistics

Examples of features include:

- `word_freq_make`
- `word_freq_address`
- `word_freq_free`
- `word_freq_money`
- `char_freq_!`
- `char_freq_$`
- `capital_run_length_average`
- `capital_run_length_longest`
- `capital_run_length_total`

The dataset contains **57 input features** and one target variable named `spam`.

### Target Variable

| Value | Class |
|---:|---|
| 0 | Non-Spam |
| 1 | Spam |

## Dataset Size

The original dataset contains:

- 4,601 observations
- 57 features
- 1 target variable

During data preprocessing, duplicate records and conflicting duplicate records were investigated and removed.

The final dataset used for modelling contains:

- **4,204 observations**
- **57 features**
- **1 target variable**

### Final Class Distribution

| Class | Count | Percentage |
|---|---:|---:|
| Non-Spam (0) | 2,528 | 60.13% |
| Spam (1) | 1,676 | 39.87% |
| **Total** | **4,204** | **100%** |

The final dataset therefore contains both classes with a moderate class imbalance.

## Train-Test Split

The final dataset was divided into training and testing datasets.

| Dataset | Observations | Features |
|---|---:|---:|
| Training | 3,363 | 57 |
| Testing | 841 | 57 |

The target distribution was maintained at approximately the same proportion in both training and testing datasets.

---

# 3. GitHub Repository Link

The complete source code, Jupyter Notebook, saved models, test data, Streamlit application, and dependency file are available in the GitHub repository:

**GitHub Repository:**

https://github.com/preetampalwe/ML-Assignment-2

The repository contains:

- `app.py`
- `requirements.txt`
- `README.md`
- Test data
- Jupyter Notebook
- Saved model files for all five implemented models

---

# 4. Models Used

## 4.1 Logistic Regression

Logistic Regression is a supervised classification algorithm used to estimate the probability that an observation belongs to a particular class.

For binary classification, the sigmoid function is used to convert the model output into a probability.

The resulting probability is used to classify an email as Spam or Non-Spam.

---

## 4.2 Decision Tree Classifier

A Decision Tree is a supervised learning algorithm that recursively divides the dataset using feature-based decision rules.

The tree consists of:

- Root node
- Internal decision nodes
- Leaf nodes

The leaf node determines the predicted class.

---

## 4.3 K-Nearest Neighbor Classifier

K-Nearest Neighbor (KNN) is a distance-based classification algorithm.

For a new observation, the algorithm identifies the `K` closest observations in the training dataset and determines the predicted class based on the classes of those neighboring observations.

Feature scaling is important for KNN because the algorithm relies on distances between observations.

---

## 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It assumes that continuous features follow a Gaussian distribution within each class and that the features are conditionally independent given the class.

---

## 4.5 Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees.

Each tree is trained using different subsets of the training data and features. The final classification is obtained by combining the predictions from the individual trees.

Random Forest was the best-performing overall model on this dataset.

---

# 5. Evaluation Metrics

Each of the five models was evaluated using the following six metrics.

## Accuracy

Accuracy measures the proportion of correctly classified observations.

---

## AUC Score

AUC (Area Under the ROC Curve) measures how well the model distinguishes between the two classes across different classification thresholds.

A higher AUC indicates better class discrimination.

---

## Precision

Precision measures the proportion of observations predicted as Spam that were actually Spam.

---

## Recall

Recall measures the proportion of actual Spam emails that were correctly identified.

---

## F1 Score

F1 Score is the harmonic mean of Precision and Recall.

---

## Matthews Correlation Coefficient

Matthews Correlation Coefficient (MCC) is a metric for evaluating binary classification performance using all four categories of the confusion matrix: TP, TN, FP and FN.

MCC is particularly useful when the classes are not perfectly balanced.

---

# 6. Model Comparison

The following table shows the performance of all five classification models on the held-out test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 92.39% | 97.21% | 90.45% | 90.45% | 90.45% | 84.12% |
| Decision Tree | 92.27% | 91.86% | 90.66% | 89.85% | 90.25% | 83.85% |
| K-Nearest Neighbors | 91.68% | 96.54% | 90.27% | 88.66% | 89.46% | 82.59% |
| Gaussian Naive Bayes | 82.52% | 94.98% | 70.35% | **97.01%** | 81.56% | 68.81% |
| **Random Forest** | **95.36%** | **99.12%** | **95.12%** | 93.13% | **94.12%** | **90.30%** |

---

# 7. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved strong performance with **92.39% accuracy** and **97.21% AUC**, while maintaining a balanced **90.45% precision** and **90.45% recall**. It performed competitively and served as a reliable baseline model. |
| Decision Tree | Delivered solid results with **92.27% accuracy** and **90.66% precision**, but had a lower **91.86% AUC** compared to Logistic Regression, indicating weaker discrimination between classes. |
| K-Nearest Neighbors (kNN) | Reached **91.68% accuracy** and **96.54% AUC**, with **90.27% precision** and **88.66% recall**. It performed reasonably well overall, though it was slightly below Logistic Regression and Decision Tree in accuracy and F1 score. |
| Naive Bayes | Had the **highest recall (97.01%)**, but the lowest overall accuracy (**82.52%**) and relatively low precision (**70.35%**). It was excellent at detecting spam emails but produced more false positives. |
| Random Forest (Ensemble) | Achieved the best overall performance with **95.36% accuracy**, **99.12% AUC**, **95.12% precision**, **93.13% recall**, **94.12% F1 score**, and **90.30% MCC**, making it the strongest model across nearly all metrics. |
| Overall Winner for your dataset? | **Random Forest** is the overall winner because it achieved the highest performance in most evaluation metrics, including Accuracy, AUC, Precision, F1 Score, and MCC. Naive Bayes led only in Recall. |

---

# 8. Overall Winner

## 🏆 Random Forest

Based on the comparison of all six evaluation metrics, **Random Forest is the overall best-performing model for this dataset**.

Random Forest achieved the best results in five of the six evaluation metrics:

| Metric | Best Model | Score |
|---|---|---:|
| Accuracy | Random Forest | **95.36%** |
| AUC | Random Forest | **99.12%** |
| Precision | Random Forest | **95.12%** |
| Recall | Gaussian Naive Bayes | **97.01%** |
| F1 Score | Random Forest | **94.12%** |
| MCC | Random Forest | **90.30%** |

Therefore, Random Forest provides the best overall balance between correctly identifying Spam and Non-Spam emails.

Gaussian Naive Bayes has the highest Recall and may be preferable in a scenario where detecting as many Spam emails as possible is the primary objective. However, its substantially lower Precision and Accuracy make Random Forest the stronger overall model.

---

# 9. Streamlit Web Application

An interactive Streamlit web application was developed to demonstrate the trained models.

The application supports the following functionality required by the assignment.

## 9.1 Dataset Upload

The user can upload a **test dataset in CSV format**.

The application validates:

- Number of columns
- Required feature names
- Unexpected columns
- Missing values
- Numeric feature values
- Target values
- Presence of both Spam and Non-Spam classes

Only test data is uploaded to the application.

---

## 9.2 Model Selection

The application provides a model selection dropdown containing all five trained models:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest

---

## 9.3 Evaluation Metrics

For the selected model, the application displays:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

The application also displays a comparison table containing the evaluation results of all five models on the uploaded test dataset.

---

## 9.4 Confusion Matrix

The application displays a confusion matrix for the selected model.

---

## 9.5 Classification Report

The application also displays the classification report for the selected model, including Precision, Recall and F1 Score for the Non-Spam and Spam classes.

---

# 10. Live Streamlit Application

The deployed Streamlit application is available at:

**Live Application:**

https://ml-assignment-2-d6nbnmzaxyx5atmnstv4ah.streamlit.app/

The application has been deployed using **Streamlit Community Cloud**.

---

# 11. Project Structure

```text
ML-Assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── spambase.csv
│   ├── test_data.csv
│   └── invalid_test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── gaussian_naive_bayes.pkl
│   └── random_forest.pkl
│
└── notebooks/
    └── ML_Assignment_2_Spambase.ipynb
```

> **Note:** `test_data.csv` is included as the test dataset used for evaluating the trained models and demonstrating the Streamlit application.

---

# 12. Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/preetampalwe/ML-Assignment-2.git
```

Navigate to the project directory:

```bash
cd ML-Assignment-2
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application can then be opened in a web browser using the local Streamlit URL displayed in the terminal.

---

# 13. Conclusion

This project demonstrates an end-to-end machine learning classification workflow using the Spambase dataset.

Five classification algorithms were implemented and compared using six evaluation metrics.

Among the evaluated models, **Random Forest achieved the best overall performance**, with an Accuracy of **95.36%**, AUC of **99.12%**, F1 Score of **94.12%**, and MCC of **90.30%**.

Gaussian Naive Bayes achieved the highest Recall of **97.01%**, demonstrating that model selection can depend on the specific evaluation metric and application objective.

The trained models were saved and integrated into an interactive Streamlit application. The application allows users to upload test data, select different models, and view their evaluation metrics, confusion matrix, and classification report.

The complete project is available through the GitHub repository and the deployed Streamlit application linked above.