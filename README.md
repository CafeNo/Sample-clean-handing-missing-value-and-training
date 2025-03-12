# Machine Learning Classifier with Ensemble Learning

## Overview
This project implements a machine learning pipeline for event classification using various ensemble learning techniques. The classifiers used include:
- **Random Forest**
- **XGBoost**
- **Gradient Boosting**
- **Voting Classifier (Ensemble Model)**

The dataset is preprocessed, resampled to handle class imbalance, and optimized using hyperparameter tuning.

## Dataset
The dataset is assumed to be located at:
```
/kaggle/input/simple-clean/train_events.csv
```

### Features and Target Variable
- **Target Variable**: `event`
- **Features**:
  - `step`
  - `timestamp`
  - Other relevant features (excluding `series_id`, `night`, and `event`)

## Dependencies
The following Python libraries are required:
```bash
pip install pandas scikit-learn xgboost imbalanced-learn
```

## Code Breakdown
### 1. Importing Libraries
```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
```

### 2. Load and Preprocess Data
```python
file_path = '/kaggle/input/simple-clean/train_events.csv'
df = pd.read_csv(file_path)

# Extract features and target variable
y = df['event']
X = df.drop(['series_id', 'night', 'event'], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='constant', fill_value=0)
X[['step']] = imputer.fit_transform(X[['step']])

# Convert timestamp to numerical format
X['timestamp'] = pd.to_datetime(X['timestamp'], errors='coerce', utc=True)
X['timestamp'].fillna(pd.to_datetime('1970-01-01T00:00:00+00:00', utc=True), inplace=True)
X['timestamp'] = X['timestamp'].astype(int) / 10**9

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

### 3. Handle Class Imbalance
```python
smote = SMOTE(random_state=42)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
```

### 4. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
```

### 5. Train Classifiers and Hyperparameter Tuning
```python
# Initialize classifiers
rf_classifier = RandomForestClassifier(random_state=42)
xgb_classifier = XGBClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15]
}
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_classifier = grid_search.best_estimator_
```

### 6. Evaluate Classifiers
```python
classifiers = [best_rf_classifier, xgb_classifier, gb_classifier]
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy {clf.__class__.__name__}: {accuracy_score(y_test, y_pred):.2f}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
    cv_accuracy = cross_val_score(clf, X_resampled, y_resampled, cv=5, scoring='accuracy').mean()
    print(f'Cross-Validation Accuracy: {cv_accuracy:.2f}\n')
```

### 7. Ensemble Learning with Voting Classifier
```python
ensemble_classifier = VotingClassifier(estimators=[
    ('rf', best_rf_classifier),
    ('xgb', xgb_classifier),
    ('gb', gb_classifier)
], voting='soft')
ensemble_classifier.fit(X_train, y_train)
y_pred_ensemble = ensemble_classifier.predict(X_test)
print(f'Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.2f}')
print(f'Classification Report Ensemble:\n{classification_report(y_test, y_pred_ensemble)}')
```

## Results
The final ensemble model is evaluated based on:
- **Accuracy**
- **Cross-Validation Score**
- **Classification Report** (Precision, Recall, F1-score)

## Usage
To run the script, execute the following command:
```bash
python script.py
```

## Future Improvements
- Feature Engineering for improved model performance
- Hyperparameter tuning for other classifiers
- Adding more ensemble techniques

## License
This project is open-source and free to use under the MIT License.

