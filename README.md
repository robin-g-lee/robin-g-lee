# Statistics and Data Science Projects

Some projects from UCLA and UCSB.

## Machine Learning
<details>
<summary>Model Summaries</summary>
<br>
  
![68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a5a43654f4542687645564c6d774368377672325256412e706e67](https://github.com/user-attachments/assets/862dd4e0-0abb-4bd2-bd35-793c421e6ca3)
  
![image](https://github.com/user-attachments/assets/60b58690-3c37-4cdc-8fc1-6de7d2f4716f)
</details>

<details>
<summary>K-Means</summary>
<br>

![image](https://github.com/user-attachments/assets/de83aac1-a121-4423-93a4-18579cbfddb4)

![image](https://github.com/user-attachments/assets/38c91b7d-24ec-40bd-9401-886ee3405259)

**Manually:**
```
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

</details>

## Time Series
<details>
<summary>Model Summaries</summary>
<br>

![image](https://github.com/user-attachments/assets/56b8612c-711f-4224-a9db-847996f5e3c4)

</details>

<details>
<summary>GARCH Models</summary>
<br>
  
![image](https://github.com/user-attachments/assets/4b9d4d2b-03bc-4685-b410-057a1c47f95c)

</details>



