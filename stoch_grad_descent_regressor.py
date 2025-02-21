from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Replace with actual dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning grid
param_grid = {
    'sgdregressor__alpha': [0.0001, 0.001, 0.01, 0.1],
    'sgdregressor__eta0': [0.001, 0.005, 0.01, 0.02, 0.05],
    'sgdregressor__max_iter': [1000, 2000, 5000, 10000],
    'sgdregressor__penalty': ['l1', 'l2', 'elasticnet']
}

# Pipeline setup
sgd_pipeline = make_pipeline(
    StandardScaler(),
    SGDRegressor(random_state=42)
)

# Randomized search for best parameters
random_search = RandomizedSearchCV(
    estimator=sgd_pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring='neg_mean_squared_error',
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
best_sgd = random_search.best_estimator_
best_sgd.fit(X_train, y_train)

# Predictions
y_train_pred = best_sgd.predict(X_train)
y_test_pred = best_sgd.predict(X_test)

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")

# Overfitting Check
if train_rmse < test_rmse * 0.8:
    print("Potential Overfitting Detected.")
elif test_rmse < train_rmse * 0.8:
    print("Potential Underfitting Detected.")
else:
    print("Model is well-balanced.")
