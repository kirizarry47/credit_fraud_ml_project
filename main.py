# Data analysis and visualization imports
import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation library
import matplotlib.pyplot as plt  # Basic plotting library
import seaborn as sns  # Statistical visualization library
from matplotlib import gridspec  # For custom plot layouts
from sklearn.model_selection import train_test_split  # Added for data splitting
from sklearn.ensemble import RandomForestClassifier  # Added for modeling
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, 
                            matthews_corrcoef, confusion_matrix)  # Consolidated metrics import

# Load and inspect the dataset
data = pd.read_csv("creditcard.csv")  # Load CSV into pandas DataFrame

# Initial data exploration
print("Dataset Preview:")
print(data.head())  # Show first 5 rows to inspect data structure
print("\nData Summary Statistics:")
print(data.describe())  # Show statistical summary of numerical columns

# Class distribution analysis
fraud_cases = data[data['Class'] == 1]  # Filter DataFrame for fraudulent transactions
valid_cases = data[data['Class'] == 0]  # Filter DataFrame for valid transactions

# Calculate and display class imbalance metrics
fraud_ratio = len(fraud_cases)/len(valid_cases)  # Ratio of fraud to valid cases
print(f"\nClass Distribution:")
print(f"Fraud Ratio: {fraud_ratio:.6f}")  # Formatted to 6 decimal places
print(f"Fraud Cases: {len(fraud_cases)}")
print(f"Valid Transactions: {len(valid_cases)}")

# Correlation analysis
print("\nGenerating correlation matrix...")
correlation_matrix = data.corr()  # Compute pairwise correlation of columns

# Correlation visualization
plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation_matrix, 
    vmax=0.8,
    square=True,
    cmap='coolwarm',
    annot=False,
    cbar_kws={'shrink': 0.75}
)
plt.title('Feature Correlation Matrix', pad=20)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Prepare data for modeling
print("\nPreparing data for modeling...")
X = data.drop('Class', axis=1)  # Features (all columns except Class)
y = data['Class']  # Target variable

# Split data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train Random Forest classifier
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Handles class imbalance
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
print("\nModel Evaluation Metrics:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    xticklabels=['Normal', 'Fraud'], 
    yticklabels=['Normal', 'Fraud']
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()