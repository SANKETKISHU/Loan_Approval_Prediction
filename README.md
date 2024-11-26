# Loan Approval Prediction

This project uses machine learning to predict whether a loan will be approved based on the applicant's details. The model is trained on a dataset containing information about applicants and their loan application status. The final model predicts loan approval for unseen test data.

## Files

- **Loan_Approval_Prediction.py**: The main Python script used to load, preprocess, train, and evaluate the machine learning model for loan approval prediction.
- **Training Dataset.csv**: The training data used to train the model (contains applicant information and loan approval status).
- **Test Dataset.csv**: The test data used for making predictions on unseen data.
- **Sample_Submission.csv**: A template for the final submission, containing columns for `Loan_ID` and the predicted `Loan_Status`.
- **submission.csv**: The final submission file with predictions for the test data.

## Prerequisites

Ensure that the following libraries are installed:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install them using pip:

 ```sh
pip install pandas numpy seaborn matplotlib scikit-learn
 ```



## Script Overview

### 1. Data Exploration

The script starts by loading and displaying basic information about the training dataset using `head()`, `info()`, and `describe()`. It also visualizes the distribution of the `Loan_Status` column, the distribution of `ApplicantIncome`, and the comparison of `ApplicantIncome` by `Loan_Status`.

### 2. Data Visualization

The following visualizations are created:
- **Loan Status Count**: A count plot showing the distribution of loan approvals (Y/N).
- **Applicant Income Distribution**: A histogram with a KDE to visualize the distribution of `ApplicantIncome`.
- **Applicant Income by Loan Status**: A box plot comparing `ApplicantIncome` for approved and non-approved loans.

### 3. Data Preprocessing

- **Numerical Features**: Includes `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`. These features are imputed for missing values using the mean and scaled using `StandardScaler`.
- **Categorical Features**: Includes `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, and `Property_Area`. These features are imputed for missing values using the most frequent value and encoded using `OneHotEncoder`.

The script then applies a `ColumnTransformer` to combine the preprocessing pipelines for both numerical and categorical features.

### 4. Model Training

A **RandomForestClassifier** is used for training the model on the preprocessed training data. The model is evaluated using accuracy, confusion matrix, and classification report on the validation set.

### 5. Test Data Predictions and Submission

The model is used to make predictions on the test data, and the results are stored in the `submission.csv` file. The final submission contains the `Loan_ID` and the predicted `Loan_Status` (Y/N).

## Conclusion

This script provides a complete pipeline for predicting loan approval based on the applicant's data. It includes data exploration, preprocessing, model training, evaluation, and generating predictions for submission. 


