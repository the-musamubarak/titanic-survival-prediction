import pandas as pd

# Load training data
train = pd.read_csv('train.csv')

# Fill missing Age values with median (without inplace warning)
train.loc[:, 'Age'] = train['Age'].fillna(train['Age'].median())

# Fill missing Embarked values with mode (without inplace warning)
train.loc[:, 'Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Convert 'Sex' column to numeric
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})

# One-hot encode 'Embarked' column and drop first category to avoid dummy variable trap
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)

# Check your updated dataframe
print(train.head())
print(train.isnull().sum())  # Confirm no missing values left


