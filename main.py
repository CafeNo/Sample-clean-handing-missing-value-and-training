import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Assuming 'event' is the target variable and 'step', 'timestamp' are features
file_path = '/kaggle/input/simple-clean/train_events.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Extract features and target variable
y = df['event']
X = df.drop(['series_id', 'night', 'event'], axis=1)

# Handling missing values
imputer = SimpleImputer(strategy='constant', fill_value=0)
X[['step']] = imputer.fit_transform(X[['step']])

# Convert the 'timestamp' column to datetime format
X['timestamp'] = pd.to_datetime(X['timestamp'], errors='coerce', utc=True)
X['timestamp'].fillna(pd.to_datetime('1970-01-01T00:00:00+00:00', utc=True), inplace=True)
X['timestamp'] = X['timestamp'].astype(int) / 10**9

# Encode the target variable using the LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Handling Imbalance using SMOTE for oversampling and RandomUnderSampler for undersampling
smote = SMOTE(random_state=42)
rus = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize classifiers
rf_classifier = RandomForestClassifier(random_state=42)
xgb_classifier = XGBClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV for Random Forest classifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15]
}

grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_classifier = grid_search.best_estimator_

# Train classifiers
classifiers = [best_rf_classifier, xgb_classifier, gb_classifier]
for clf in classifiers:
    # Fit the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy {clf.__class__.__name__}: {accuracy:.2f}')
    print(f'Classification Report {clf.__class__.__name__}:\n{classification_report(y_test, y_pred)}')

    # Use cross-validation to get a better estimate of the model's performance
    cv_accuracy = cross_val_score(clf, X_resampled, y_resampled, cv=5, scoring='accuracy').mean()
    print(f'Cross-Validation Accuracy {clf.__class__.__name__}: {cv_accuracy:.2f}\n')

# Ensemble classifier using VotingClassifier
ensemble_classifier = VotingClassifier(estimators=[
    ('rf', best_rf_classifier),
    ('xgb', xgb_classifier),
    ('gb', gb_classifier)
], voting='soft')

# Fit the ensemble classifier
ensemble_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ensemble = ensemble_classifier.predict(X_test)

# Evaluate the performance of the ensemble classifier
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'\nAccuracy Ensemble: {accuracy_ensemble:.2f}')
print(f'Classification Report Ensemble:\n{classification_report(y_test, y_pred_ensemble)}')
