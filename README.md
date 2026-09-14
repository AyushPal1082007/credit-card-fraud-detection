# Credit Card Fraud Detection using XGBoost

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning.

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small fraction of all transactions.

The project uses **XGBoost** as the primary machine learning model and compares its performance with an **SVM baseline**.

The project also uses **SMOTE** to handle class imbalance and **decision threshold tuning** to improve the balance between precision and recall.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze a highly imbalanced credit card transaction dataset.
* Identify fraudulent and legitimate transactions.
* Handle class imbalance using SMOTE.
* Build an XGBoost fraud detection model.
* Build an SVM model as a baseline.
* Compare SVM and XGBoost performance.
* Tune the decision threshold for fraud classification.
* Evaluate the models using appropriate classification metrics.
* Interpret the XGBoost model using feature importance scores.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset**.

The dataset contains:

* `Time` — Time elapsed between transactions.
* `V1` to `V28` — PCA-transformed transaction features.
* `Amount` — Transaction amount.
* `Class` — Target variable.

Target variable:

```text
0 = Normal transaction
1 = Fraudulent transaction
```

The dataset is highly imbalanced, with fraudulent transactions representing only a very small percentage of the total transactions.

---

## 🛠️ Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Imbalanced-learn
* XGBoost

---

## 🤖 Machine Learning Models

### 1. SVM

A Linear SVM was used as the baseline model.

The SVM was configured with class balancing to account for the imbalanced nature of the dataset.

### 2. XGBoost

XGBoost was used as the primary classification model.

The model was trained after applying SMOTE to the training data.

---

## ⚖️ Handling Class Imbalance

The dataset contains significantly more legitimate transactions than fraudulent transactions.

SMOTE was applied only to the training dataset:

```text
Original Training Data
        ↓
      SMOTE
        ↓
Balanced Training Data
        ↓
     XGBoost
```

Validation and test data were kept untouched to prevent data leakage.

---

## 🔀 Data Splitting

The dataset was divided into:

* 60% Training
* 20% Validation
* 20% Testing

Stratified sampling was used to preserve the class distribution.

---

## 📏 Evaluation Metrics

Because the dataset is highly imbalanced, accuracy was not used as the primary evaluation measure.

The following metrics were used:

### ROC-AUC

Measures how well the model separates fraudulent and legitimate transactions across different thresholds.

### PR-AUC

Measures the precision-recall performance and is particularly useful for highly imbalanced datasets.

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

Used to analyze:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 🎚️ Decision Threshold Tuning

Instead of using the default classification threshold of `0.50`, multiple thresholds were tested using the validation dataset.

The threshold producing the highest F1-score was selected.

The selected threshold was then applied to the untouched test dataset.

This approach is useful in fraud detection because the cost of missing fraudulent transactions can be high.

---

## 🔍 Feature Importance

XGBoost feature importance scores were extracted to identify the features that contributed most to the model's predictions.

The top 15 features were visualized using a bar chart.

This provides a basic level of interpretability for the fraud detection model.

---

## 📈 Project Workflow

```text
Credit Card Dataset
        ↓
Data Exploration
        ↓
Remove Duplicate Records
        ↓
Train / Validation / Test Split
        ↓
Feature Scaling
        ↓
       SVM
     Baseline
        ↓
    Evaluation
        ↓
SMOTE on Training Data
        ↓
     XGBoost
        ↓
    Evaluation
        ↓
Model Comparison
        ↓
Threshold Tuning
        ↓
Final Test Evaluation
        ↓
Confusion Matrix
        ↓
Feature Importance
```

---

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

1. Open the notebook in Google Colab.
2. Upload the `creditcard.csv` dataset.
3. Run the notebook cells in order.
4. Install the required Python libraries.
5. Train the SVM baseline.
6. Apply SMOTE to the training data.
7. Train the XGBoost model.
8. Perform threshold tuning.
9. Evaluate the final model.
10. View the feature importance results.

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
```

---

## ✅ Results

The notebook provides a comparison between SVM and XGBoost using:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-Score

The final results depend on the model execution and selected decision threshold.

The notebook also provides:

* ROC curve
* Precision-Recall curve
* Threshold tuning graph
* Confusion matrix
* XGBoost feature importance graph

---

## 🧠 Key Takeaways

* Credit card fraud detection is a severely imbalanced classification problem.
* Accuracy alone can be misleading for this type of dataset.
* SMOTE can help the model learn from the minority fraud class.
* SMOTE should only be applied to the training data.
* XGBoost provides a powerful approach for detecting fraudulent transactions.
* Decision threshold tuning can improve the balance between precision and recall.
* Feature importance provides insight into which variables influence the model's predictions.

---

## 👨‍💻 Project

**Project:** Credit Card Fraud Detection
**Primary Model:** XGBoost
**Baseline Model:** SVM
**Imbalance Technique:** SMOTE
**Evaluation:** ROC-AUC, PR-AUC, Precision, Recall, F1-Score, Confusion Matrix
