# titanic_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Load Data ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

# === Handle Missing Data ===
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# === Encode Categorical Data ===
for df in [train, test]:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# === Features and Target ===
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
y = train['Survived']
X_test = test[features]

# === Split for Validation ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Training ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Validation ===
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# === Predict on Test Set ===
predictions = model.predict(X_test)

# === Save Submission ===
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

# === Plot Saving Setup ===
os.makedirs("images", exist_ok=True)

# === Plot 1: Survival Rate by Gender ===
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=train)
plt.title("Survival Rate by Gender")
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.tight_layout()
plot1_path = "images/survival_by_gender.png"
plt.savefig(plot1_path, dpi=300)
plt.close()

# === Plot 2: Feature Importance ===
importances = model.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df, palette='viridis')
plt.title("Feature Importance")
plt.tight_layout()
plot2_path = "images/feature_importance.png"
plt.savefig(plot2_path, dpi=300)
plt.close()

# === Save Image Log to CSV ===
image_log = pd.DataFrame([
    {"Image": plot1_path, "Description": "Survival Rate by Gender"},
    {"Image": plot2_path, "Description": "Feature Importance"}
])
image_log.to_csv("image_summary.csv", index=False)
print("Images saved and image_summary.csv created.")



