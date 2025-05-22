import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# -------------------------------
# Feature Engineering

# Extract Title from Name
train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Simplify Titles
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
    'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare', 'Jonkheer': 'Rare',
    'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
}

train['Title'] = train['Title'].map(title_mapping).fillna('Rare')
test['Title'] = test['Title'].map(title_mapping).fillna('Rare')

# Fill missing Age with median by Title
train['Age'] = train.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

# Fill missing Fare in test set with median Fare
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Fill missing Embarked in train set with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Convert categorical to numeric
sex_map = {'male': 0, 'female': 1}
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}

train['Sex'] = train['Sex'].map(sex_map)
test['Sex'] = test['Sex'].map(sex_map)

train['Embarked'] = train['Embarked'].map(embarked_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

train['Title'] = train['Title'].map(title_map)
test['Title'] = test['Title'].map(title_map)

# Drop columns not used in modeling
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
X_train = train.drop(columns=drop_cols + ['Survived'])
y_train = train['Survived']
X_test = test.drop(columns=drop_cols)

# -------------------------------
# Split training data for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# -------------------------------
# Hyperparameter tuning for RandomForestClassifier

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_tr, y_tr)

print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation accuracy: {random_search.best_score_}")

best_rf = random_search.best_estimator_

# Evaluate on validation set
val_preds = best_rf.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy with tuned model: {val_accuracy}")

# -------------------------------
# Feature importance

importances = best_rf.feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save feature importance plot
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df, palette='viridis', dodge=False)
plt.title('Feature Importance\nAuthor: the-musamubarak (GitHub)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.legend(['Random Forest Feature Importance'])
plt.tight_layout()
plt.savefig('feature_importance_updated.png')
print("Feature importance plot saved as feature_importance_updated.png")
plt.close()

# -------------------------------
# Additional visualizations

# 1. Survival Rate by Title
plt.figure(figsize=(10,6))
title_survival = train.groupby('Title')['Survived'].mean().sort_values()
sns.barplot(x=title_survival.index, y=title_survival.values, palette='coolwarm')
plt.title('Survival Rate by Title\nAuthor: the-musamubarak (GitHub)', fontsize=14)
plt.xlabel('Title')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('survival_by_title.png')
print("Survival by Title plot saved as survival_by_title.png")
plt.close()

# 2. Confusion Matrix on Validation Set
cm = confusion_matrix(y_val, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did Not Survive', 'Survived'])
plt.figure(figsize=(6,6))
disp.plot(cmap='Blues', ax=plt.gca(), colorbar=False)
plt.title('Confusion Matrix on Validation Set\nAuthor: the-musamubarak (GitHub)', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as confusion_matrix.png")
plt.close()

# -------------------------------
# Predict on test set and save submission

test_preds = best_rf.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")





