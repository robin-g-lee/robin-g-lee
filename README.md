# Projects and Notes

- Repositories contain old projects while at UCLA and UCSB.
- See below for notes I use at work/school.

<br>

## Machine Learning Notes

Feature Selection: https://www.stratascratch.com/blog/feature-selection-techniques-in-machine-learning/

![image](https://github.com/user-attachments/assets/6c33f351-39d1-4f7f-a06a-923725c028de)

`LightFM` is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.

> https://making.lyst.com/lightfm/docs/home.html#quickstart

`Surprise` is a Python scikit for building and analyzing recommender systems that deal with explicit rating data.

> https://surpriselib.com/

<br>

<details>
<summary>Clustering: K-Means, DBSCAN</summary>
<br>

### K-Means:

Clustering seeks to find N clusters in a data set and to subsequently identify which data points belong to each cluster. While there are a number of different approaches to clustering, one of the easiest to understand is the k-means algorithm. 

In this algorithm:

1. Pick K random points as cluster centers called centroids.
2. Assign each point to the nearest cluster by calculating its Euclidean distance to each centroid.
3. Find a new cluster center by taking the average of the assigned points.
4. Repeat Step 2 and 3 until none of the cluster assignments change.

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

<br>

</details>

<details>
<summary>CART and Ensemble Learning</summary>
<br>

### Classification And Regression Tree (CART)

* Trees used for regression and trees used for classification have some similarities - but also some differences, such as the procedure used to determine where to split.

* Some techniques, often called ensemble methods, construct more than one decision tree:
  
  * **Boosted trees:** Incrementally building an ensemble by training each new instance to emphasize the training instances previously mis-modeled. A typical example is AdaBoost. These can be used for regression-type and classification-type problems.
  * **Bagging:** Bootstrap aggregated (or bagged) decision trees, an early ensemble method, builds multiple decision trees by repeatedly resampling training data with replacement, and voting the trees for a consensus prediction.


### Limitations of CARTs:

* **Diagnose Variance Problems**

    * If $\hat{f}$ suffers from **high variance**:
      </br><mark>CV error of $\hat{f}$ > training set error of $\hat{f}$</mark>

    * $\hat{f}$ is said to overfit the training set. To remedy overfitting: **decrease**
        * model complexity,
        * i.e. decrease max depth, increase min samples per leaf, ...
        * gather more data, ..
 
* **Diagnose Variance Problems**

    * If $\hat{f}$ suffers from **high bias**:
    </br><mark>CV error of $\hat{f} \approx$ training set error of $\hat{f}$ >> desired error.</mark>

    * $\hat{f}$ is said to underfit the training set. To remedy underfitting: **increase**
        * model complexity
        * i.e. increase max depth, decrease min samples per leaf, ...
        * gather more relevant features


### Ensemble Learning:

* **Bagging:** Bootstrap Aggregation.
  * Base estimator: Decision Tree, Logistic Regression, Neural Net, ...
  * Each estimator is trained on a distinct bootstrap sample of the training set

    ![image](https://github.com/user-attachments/assets/abbb0112-8143-4649-8794-0b57924aa97e)

* **Boosting:** several models are trained sequentially with each model learning from the errors of its predecessors
  * AdaBoost and Gradient Boosting
    
    ![image](https://github.com/user-attachments/assets/6ca714aa-dcec-4947-94a5-220375a57450)


### Examples:

* CART:

  <details>
  <summary><strong>Decision Tree Regressor</strong></summary>
  <br>
  
  **Steps:**

  1. **Data Preparation**
  
  2. **Selecting the Root Node**
      * Choose the best feature: Select the feature that provides the most information gain or best splits the data based on a chosen metric like Gini impurity or entropy.
      * Create the root node: This becomes the starting point of the decision tree.
    
  3. **Splitting the Data**
      * Create child nodes: Based on the chosen feature, split the data into multiple branches representing different possible values of that feature.
      * For each potential split, calculate the information gain or impurity reduction to select the best split point.
    
  4. **Recursive Tree Building:**
      * For each child node, repeat the steps of selecting the best feature and splitting the data further, creating new child nodes until a stopping criterion is met.
     
  5. **Stopping Criteria:**
      * Maximum depth: Limit the number of levels in the tree to prevent overfitting.
      * Minimum sample size: Stop splitting when a node contains too few data points.
      * Pre-defined accuracy threshold: Stop when the model reaches a desired level of accuracy.
    
  6. **Leaf Nodes:**
      * Assign predictions: At the end of each branch (leaf node), assign a prediction based on the majority class label for classification problems or the average value for regression problems. 

  <br>

  **Disadvantages of decision trees:**
  * **Overfitting:** Decision trees can easily overfit to training data, meaning they perform well on the training set but poorly on new data due to complex decision rules.
  * **Sensitivity to data changes:** Small changes in the training data can lead to significantly different decision tree structures.
  * **Greedy approach:** The algorithm chooses the best split at each node locally, which may not lead to the globally optimal decision tree

  </details>

* Boosting:

  <details>
  <summary><strong>Gradient Boosting Regressor</strong></summary>
  <br>

  The model’s strength comes from its additive learning process — while each tree focuses on correcting the remaining errors in the ensemble, the sequential combination creates a powerful predictor that progressively reduces the overall
  prediction error by focusing on the parts of the problem where the model still struggles.

  1. **Initialize Model:** Start with a simple prediction, typically the mean of target values.
  2. **Iterative Learning:** For a set number of iterations, compute the residuals, train a decision tree to predict these residuals, and add the new tree’s predictions (scaled by the learning rate) to the running total.
  3. **Build Trees on Residuals:** Each new tree focuses on the remaining errors from all previous iterations.
  4. **Final Prediction:** Sum up all tree contributions (scaled by the learning rate) and the initial prediction.

  <br>
  
  **Risk of Overfitting:** The use of deeper trees and the sequential building process can cause the model to fit the training data too closely, which may reduce its performance on new data. This requires careful tuning of tree depth, learning rate, and the number of trees.
    
  **Sensitive to Settings:** The effectiveness of Gradient Boosting heavily depends on finding the right combination of learning rate, tree depth, and number of trees, which can be more complex and time-consuming than tuning simpler algorithms.

  ![image](https://github.com/user-attachments/assets/e416e064-838e-44bb-8e09-b0cd36b1bfc6)

  https://medium.com/towards-data-science/gradient-boosting-regressor-explained-a-visual-guide-with-code-examples-c098d1ae425c

  </details>
  
* Bagging:
  
  <details>
  <summary><strong>Random Forest Regressor</strong></summary>
  <br>

  Why Random Forests Work:

  **Variance reduction:** the trees are more independent because of the combination of bootstrap samples and random draws of predictors.
  It is apparent that random forests are a form of bagging, and the averaging over trees can substantially reduce instability that might otherwise result.
  Moreover, by working with a random sample of predictors at each possible split, the fitted values across trees are more independent.
  Consequently, the gains from averaging over a large number of trees (variance reduction) can be more dramatic.

  **Bias reduction:** a very large number of predictors can be considered, and local feature predictors can play a role in tree construction.

  **Cons:** computational complexity, slower performance compared to simpler models, and lack of interpretability

    
  ![image](https://github.com/user-attachments/assets/239845b7-9dce-4df5-9a67-86b0697da2c1)

  
  </details>

  <br>

</details>


<details>
<summary>SHAP and LIME Model Interpretability</summary>
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

<br>
  
## Time Series Notes

![image](https://github.com/user-attachments/assets/56b8612c-711f-4224-a9db-847996f5e3c4)


<details>
<summary>GARCH Models</summary>
<br>
  
![image](https://github.com/user-attachments/assets/4b9d4d2b-03bc-4685-b410-057a1c47f95c)

https://medium.com/@corredaniel1500/forecasting-volatility-deep-dive-into-arch-garch-models-46cd1945872b

</details>
