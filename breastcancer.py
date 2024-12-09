
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


data = pd.read_csv("/content/BreastCancer (1).csv")  
print("Data loaded successfully!")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

sns.countplot(x='diagnosis', data=data)
plt.title('diagnosis Count')
plt.show()

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0}) 

data['diagnosis'].fillna(data['diagnosis'].mode()[0], inplace=True)


data.fillna(data.mean(), inplace=True)


if 'id' in data.columns:
    data = data.drop(['id'], axis=1)
if 'Unnamed: 32' in data.columns:
    data = data.drop(['Unnamed: 32'], axis=1)

if len(data.select_dtypes(exclude=[np.number]).columns) > 0:
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    print("Non-numeric columns:", non_numeric_cols)
    data = pd.get_dummies(data, columns=non_numeric_cols)


X = data.drop(['diagnosis'], axis=1)  
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nData Split:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)  

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)


vc = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
vc.fit(X_train, y_train)


models = {'Random Forest': rf, 'Gradient Boosting': gb, 'Voting Classifier': vc}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Model")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

import pickle
filename = 'breast_cancer_model.pkl'
