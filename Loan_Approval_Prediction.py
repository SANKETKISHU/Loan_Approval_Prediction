import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
train_data = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_8/Training Dataset.csv')
test_data = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_8/Test Dataset.csv')
sample_submission = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_8/Sample_Submission.csv')

# Data Exploration
print(train_data.head())
print(train_data.info())
print(train_data.describe())

# Data Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='Loan_Status', data=train_data)
plt.title('Loan Status Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_data['ApplicantIncome'], kde=True)
plt.title('Distribution of Applicant Income')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=train_data)
plt.title('Applicant Income by Loan Status')
plt.show()

# Data Preprocessing
# Define numerical and categorical columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Create preprocessing pipelines for both numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

# Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Split data into features and target
X = train_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_data['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor on the training data
preprocessor.fit(X_train)

# Transform the training and validation data
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_processed, y_train)

# Evaluation
y_pred = model.predict(X_val_processed)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Preparing test data and making predictions
X_test = test_data.drop(columns=['Loan_ID'])
X_test_processed = preprocessor.transform(X_test)
test_predictions = model.predict(X_test_processed)

# Prepare submission file
submission = pd.DataFrame({'Loan_ID': test_data['Loan_ID'], 'Loan_Status': ['Y' if pred == 1 else 'N' for pred in test_predictions]})
submission.to_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_8/submission.csv', index=False)
