# Financial Fraud Detection using Machine Learning

## Overview

This project detects fraudulent financial transactions using Machine Learning techniques. The model is trained on over 6.3 million real-world transaction records and predicts whether a transaction is fraudulent.

---

## Dataset

- Transactions: 6,362,620
- Features: 16
- Fraud Cases: 8,213
- Fraud Rate: 0.129%

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- VS Code

---

## Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Visualization
4. Feature Engineering
5. Data Preprocessing
6. Random Forest Model
7. XGBoost Model
8. Feature Importance
9. Model Saving
10. Fraud Prediction
11. Model Comparison
12. Data Leakage Check
13. Cross Validation

---

## Feature Engineering

Created new features:

- Sender Balance Difference
- Receiver Balance Difference
- Sender Account Emptied
- Amount to Balance Ratio
- High Value Transaction
- Transaction Hour
- Transaction Day

---

## Model Performance

### Random Forest

- Accuracy: 100%
- Precision: 100%
- Recall: 98.45%
- F1 Score: 99.22%

### XGBoost

- Accuracy: 100%
- Precision: 98.44%
- Recall: 97.67%
- F1 Score: 98.05%

Random Forest was selected as the final model.

---

## Cross Validation

Average F1 Score: **99.26%**

---

## Project Structure

```
Financial_Fraud_Detection_ML
│
├── Data
├── Models
├── Reports
├── Python
├── SQL
├── Tableau
└── README.md
```

---

## Future Improvements

- Real-time fraud detection
- Streamlit Web Application
- REST API Deployment
- Cloud Deployment
- Deep Learning Models

---

## Author

Harshita Shrotriy
