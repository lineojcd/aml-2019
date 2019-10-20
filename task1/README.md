# aml-2019
Repo for Advanced Machine Learning at ETHZ Autumn Semester 2019

## PCA

`n_features = [2, 5, 10, 12, 15, 17, 19, 20, 50, 75, 100, 150, 200]`

### Ridge

Model: `sklearn.linear_model.Ridge(alpha=alpha)`
Best: `alpha=1, n_features=12`

### RandomForestRegressor

Model: `sklearn.linear_model.Ridge(alpha=alpha)`
No significant difference between 300 and 500 `n_estimators` and 10, 12 features seems to work best