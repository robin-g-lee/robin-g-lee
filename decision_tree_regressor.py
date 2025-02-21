# Hyperparameter Turning
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
param_dist = {
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'criterion': ['mse', 'friedman_mse', 'mae']
}
random_search = RandomizedSearchCV(
    DecisionTreeRegressor(),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(train_X, train_y)
  
# Get the best parameters
print("Best hyperparameters found: ", random_search.best_params_)

# Train the model
decision_tree = DecisionTreeRegressor(max_depth=18,
                                    min_samples_leaf=15,
                                    min_samples_split=3,
                                    random_state=3,
                                   criterion='friedman_mse')
decision_tree.fit(train_X, train_y)

## Test overfitting or underfitting
from sklearn.metrics import mean_squared_error as MSE
# Compute y_pred
y_pred = decision_tree.predict(test_X)
# Compute mse_dt
mse_dt = MSE(test_y, y_pred)
# Compute rmse_dt
rmse_dt = mse_dt**(1/2)
# Print rmse_dt
print("Test set RMSE of decision_tree: {:.4f}".format(rmse_dt))

from sklearn.model_selection import cross_val_score
# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(decision_tree, train_X, train_y, cv=10, 
                                scoring='neg_mean_squared_error', 
                                n_jobs=-1) 
# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)
# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
# Fit dt to the training set
decision_tree.fit(train_X, train_y)
# Predict the labels of the training set
y_pred_train = decision_tree.predict(train_X)
# Evaluate the training set RMSE of dt
RMSE_train = (MSE(train_y, y_pred_train))**(1/2)
# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

# See if Over or Under Fitting
