from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Replace with actual dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning grid
param_grid = {
    'linearsvc__C': [0.01, 0.1, 1, 10, 100],
    'linearsvc__loss': ['hinge', 'squared_hinge'],
    'linearsvc__max_iter': [1000, 5000, 10000]
}

# Pipeline with feature scaling
svc_pipeline = make_pipeline(
    StandardScaler(),
    LinearSVC(random_state=42, dual=False)
)

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=svc_pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

# Best hyperparameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Train the best model
best_svc = random_search.best_estimator_
best_svc.fit(X_train, y_train)

# Predictions
y_train_pred = best_svc.predict(X_train)
y_test_pred = best_svc.predict(X_test)

# Evaluation
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion Matrix & Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Overfitting Check
if train_acc > test_acc + 0.1:
    print("Potential Overfitting Detected.")
elif test_acc > train_acc + 0.1:
    print("Potential Underfitting Detected.")
else:
    print("Model is well-balanced.")
