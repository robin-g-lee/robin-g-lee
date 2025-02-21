# Statistics and Data Science Projects

Some projects from UCLA and UCSB.


## Machine Learning Notes

Algorithms: https://github.com/topics/machine-learning-algorithms

Feature Selection: https://www.stratascratch.com/blog/feature-selection-techniques-in-machine-learning/

![image](https://github.com/user-attachments/assets/6c33f351-39d1-4f7f-a06a-923725c028de)

`LightFM` is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.

> https://making.lyst.com/lightfm/docs/home.html#quickstart

`Surprise` is a Python scikit for building and analyzing recommender systems that deal with explicit rating data.

> https://surpriselib.com/

<details>
<summary>Clustering: K-Means, DBSCAN</summary>
<br>

### K-Means:

![image](https://github.com/user-attachments/assets/de83aac1-a121-4423-93a4-18579cbfddb4)

![image](https://github.com/user-attachments/assets/38c91b7d-24ec-40bd-9401-886ee3405259)

```
## Manually:
# Euclidean Distance Calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
k = 3 # Number of clusters
C_x = np.random.randint(0, np.max(X)-20, size=k) # random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k) # random centroids
C = np.array(list(zip(C_x, C_y)), dtype=np.float32) # sample data

C_old = np.zeros(C.shape) # store the value of centroids when it updates
clusters = np.zeros(len(X)) # creates Cluster Lables(0, 1, 2)
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
while error != 0: # Loop will run till the error becomes zero
    for i in range(len(X)): # Assigning each value to its closest cluster
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C) # Storing the old centroid values
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
```

### DBSCAN:

How DBSCAN works:
1. Groups points that are close together based on density
2. Marks points that are alone in low-density regions as outliers
3. Defines clusters as dense regions separated by regions of lower density 

DBSCAN's advantages:
1. Doesn't require the number of clusters to be specified beforehand
2. Can identify clusters of arbitrary shapes
3. Effective at identifying and removing noise in a data set
4. Robust to noise

```
from sklearn.preprocessing import StandardScaler
import numpy as np
data = pd.read_csv('Week_2_Data/Iris.csv')
data = data[['SepalLengthCm','PetalLengthCm','Species']]
species = data['Species']
features = data.iloc[:, :-1]
features = features.values.astype("float32", copy = False)

stscaler = StandardScaler().fit(features)
features = stscaler.transform(features)

# Elbow plot:
k=2 # number of columns
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=k+1).fit(features) 
distances, indices = nbrs.kneighbors(features)
distances = np.sort(distances[:, k])

plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("k-NN distance")
plt.title("k-NN Distance Plot")
plt.show()
# min_samples is 12, because first jump (knee) is between 10 and 15
# eps is 0.5

dbsc = DBSCAN(eps = .5, min_samples = 12).fit(features)
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True
data['label'] = labels
print(data['label'].unique())
print(data.head())

plt.figure(figsize=(8, 6))
# Plot the clusters
colors = ['indianred','#57db5f','#5f57db']
for label in data['label'].unique():
    cluster_data = data[data['label'] == label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['PetalLengthCm'], 
                label=f'Class {label}',
                color = colors[label % len(colors)],
               s = 150)
plt.legend()
plt.title("Iris Data")
```


<br>

</details>

<details>
<summary>CART and Ensemble Learning</summary>
<br>

### Classification And Regression Tree (CART)

* Trees used for regression and trees used for classification have some similarities - but also some differences, such as the procedure used to determine where to split (!)

* Some techniques, often called ensemble methods, construct more than one decision tree:
  * **Boosted trees:** Incrementally building an ensemble by training each new instance to emphasize the training instances previously mis-modeled. A typical example is AdaBoost. These can be used for regression-type and classification-type problems.
  * **Bagging:** Bootstrap aggregated (or bagged) decision trees, an early ensemble method, builds multiple decision trees by repeatedly resampling training data with replacement, and voting the trees for a consensus prediction.
  * Random forest classifier is a specific type of bootstrap aggregating
  * Rotation forest.


### Limitations of CARTs:

![image](https://github.com/user-attachments/assets/1a027e3e-6e19-4c0c-b4f8-0e5b9a5d74bb)
![image](https://github.com/user-attachments/assets/fa891e83-d007-4516-9696-b03bc32e014e)

### Ensemble Learning:

* **Bagging:** Bootstrap Aggregation.
  * Base estimator: Decision Tree, Logistic Regression, Neural Net, ...
  * Each estimator is trained on a distinct bootstrap sample of the training set

![image](https://github.com/user-attachments/assets/a348f332-7224-4ce3-8d1a-27f6ae00f35a)

* **Boosting:** several models are trained sequentially with each model learning from the errors of its predecessors
  * AdaBoost and Gradient Boosting
    
![image](https://github.com/user-attachments/assets/01023cae-a774-4164-898f-2b0474fc68d2)


### Examples of Ensemble:
  
* Bagging:
  
  <details>
  <summary><strong>Random Forest</strong></summary>
  <br>

  Why Random Forests Work: Variance reduction: the trees are more independent because of the combination of bootstrap samples and random draws of predictors.
  It is apparent that random forests are a form of bagging, and the averaging over trees can substantially reduce instability that might otherwise result.
  Moreover, by working with a random sample of predictors at each possible split, the fitted values across trees are more independent.
  Consequently, the gains from averaging over a large number of trees (variance reduction) can be more dramatic.
  Bias reduction: a very large number of predictors can be considered, and local feature predictors can play a role in tree construction.

    
  ![image](https://github.com/user-attachments/assets/c86b06bf-da91-4fd5-a5e3-11c0fb6bde2e)
  
  ```
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import RandomizedSearchCV
  from scipy.stats import randint
  param_dist = {
      'n_estimators': randint(100, 1000),
      'max_depth': randint(1, 20),
      'min_samples_split': randint(2, 20),
      'min_samples_leaf': randint(1, 20),
      'max_features': ['auto', 'sqrt', 'log2', None],
      'criterion': ['absolute_error', 'poisson', 'friedman_mse', 'squared_error'],
      'bootstrap': [True, False]
  }
  random_search = RandomizedSearchCV(
      RandomForestRegressor(),
      param_distributions=param_dist,
      n_iter=100,
      cv=5,
      verbose=1,
      random_state=42,
      n_jobs=-1
  )
  random_search.fit(train_X, train_y)
  print("Best hyperparameters found: ", random_search.best_params_)
  
  from sklearn.ensemble import RandomForestRegressor
  rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                                        min_samples_leaf=10,
                                        min_samples_split=5, random_state=42)
  rf.fit(train_X,train_y)
  
  from sklearn.metrics import mean_squared_error as MSE
  y_pred = rf.predict(test_X)
  y_pred_train=rf.predict(train_X)
  # Evaluate the test set RMSE
  rmse_test = MSE(test_y, y_pred)**(1/2)
  rmse_train = MSE(train_y, y_pred_train)**(1/2)
  # Print the test set RMSE
  print('Test set RMSE of rf: {:.3f}'.format(rmse_test))
  print('Train set RMSE of rf: {:.3f}'.format(rmse_train))
  
  from sklearn.model_selection import cross_val_score
  # Compute the array containing the 10-folds CV MSEs
  MSE_CV_scores = - cross_val_score(rf, train_X, train_y, cv=10, 
                                    scoring='neg_mean_squared_error', 
                                    n_jobs=-1) 
  # Compute the 10-folds CV RMSE
  RMSE_CV = (MSE_CV_scores.mean())**(1/2)
  # Print RMSE_CV
  print('CV RMSE: {:.2f}'.format(RMSE_CV))
  
  y_pred_rf = rf.predict(test_X)
  rmse_rf = np.sqrt(mean_squared_error(test_y, y_pred_rf))
  print('RMSE (Random Forest): ', rmse_rf)
  ```
  
  </details>

  <details>
  <summary><strong>Bagging Classifier</strong></summary>
  <br>

  ```
  from sklearn.ensemble import BaggingClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import train_test_split
  # Set seed for reproducibility
  SEED = 1
  # Split data into train and test
  X_train, X_test, y_train, y_test = \
  train_test_split(X, y,
  test_size=0.2,
  stratify=y,
  random_state=SEED)

  # Hyperparameter Turning
  # Insert here for decision tree classifier

  # Instantiate a classification-tree 'dt'. Can use tuning.
  dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16,
  random_state=SEED)
  # Instantiate a BaggingClassifier 'bc'
  bc = BaggingClassifier(base_estimator=dt, n_estimators=300,
  n_jobs=-1)
  # Fit 'bc' to the training set
  bc.fit(X_train, y_train)
  # Predict test set labels
  y_pred = bc.predict(X_test)
  # Evaluate and print test-set accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))
  ```

  </details>

  <br>

</details>



<details>
<summary>Model Interpretability: SHAP and LIME</summary>
<br>
  
https://medium.com/cmotions/opening-the-black-box-of-machine-learning-models-shap-vs-lime-for-model-explanation-d7bf545ce15f
  
### SHAP: SHapley Additive exPlanations

This method aims to explain the prediction of an instance/observation by computing the contribution of each feature to the prediction. Uses game theory to explain a model by considering each feature as a player. SHAP values are relative to the average predicted value of the sample.

https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html

Supervised Clustering: How to Use SHAP Values for Better Cluster Analysis:

https://www.aidancooper.co.uk/supervised-clustering-shap-values/#:~:text=Clustering%20the%202D%20Embedding%20of,we%20elect%20to%20use%20DBSCAN.&text=As%20expected%2C%20this%20identifies%20the,discerned%20visually%20(Figure%205)

### LIME: Local Interpretable Model-Agnostic Explanations

Approximates a complex model and transfers it to a local interpretable model. LIME generates a perturbed dataset to fit an explainable model.

https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html

|  Step  |   Description                                                                       |
|:-------|:------------------------------------------------------------------------------------|
|Let     | Let’s say we want to know why the model predicted that someone earns more than $50K |
|Change  | Change the Example a Little Bit</br> LIME makes small changes to data (increasing age, changing job, or reducing education level).</br> It asks the model, “What happens now? |
|Find Out| Find Out Which Changes Matter</br> If changing job causes the prediction to flip (now the model says they earns less), then job is very important!</br> If changing age doesn’t affect the prediction much, then age is not very important |
|Make    | Make a Simple Explanation</br> LIME builds a small, simple model (like drawing a straight line) to explain what’s happening just around person's case.</br> It tells you which features (age, job, education, etc.) were the most important for this one prediction |
| LIME   | LIME only explains one example at a time (not the whole model).</br> LIME makes fake, small changes to see what affects the decision.</br> LIME creates a simple explanation (even if the original model is very complex).|

![image](https://github.com/user-attachments/assets/535c9217-b17e-48e5-a8ce-d6b7e95b057c)

https://medium.com/towards-data-science/lime-explain-machine-learning-predictions-af8f18189bfe

### Comparison

![image](https://github.com/user-attachments/assets/02449983-b8c3-4296-a71f-d0209d1dbf34)

![image](https://github.com/user-attachments/assets/27a67997-93ad-481c-bf12-28ed5d33036a)

<br>

</details>


  
## Time Series Notes

![image](https://github.com/user-attachments/assets/56b8612c-711f-4224-a9db-847996f5e3c4)


<details>
<summary>GARCH Models</summary>
<br>
  
![image](https://github.com/user-attachments/assets/4b9d4d2b-03bc-4685-b410-057a1c47f95c)

https://medium.com/@corredaniel1500/forecasting-volatility-deep-dive-into-arch-garch-models-46cd1945872b

</details>
