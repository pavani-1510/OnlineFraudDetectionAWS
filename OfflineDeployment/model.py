# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
# Load dataset
df = pd.read_csv('/home/pavani-r/Desktop/CBAIMLS/Project/onlinefraud.csv')
# Import necessary libraries


print(df.isnull().sum())

# Step 3: Handling Missing Values (if any)
df = df.dropna()

# Step 4: Encode Categorical Variables
# The 'type' column is categorical and needs to be encoded
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])  # Convert 'type' column to numerical

print("\nUnique values in 'type' column after encoding:", df['type'].unique())

# Step 5: Feature Selection
# Dropping irrelevant columns like 'nameOrig' and 'nameDest' (non-numerical, unique identifiers)
X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
y = df['isFraud']

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nShape of Training and Testing Sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Model Building
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 9: Model Prediction
y_pred = model.predict(X_test_scaled)

# Step 10: Model Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 11: Save the model for deployment
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the scaler as well
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("\nModel and Scaler saved successfully!")
