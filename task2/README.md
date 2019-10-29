# task1 

Disease Classification

## Under-sampling vs over-sampling

### Over-sampling

```python
('scaler', StandardScaler()),
('smote', SMOTE()),
('clf', SVM()),   
```

clf with default parameters

| Score           | clf |
|-----------------|-----|
| 0.634 +/- 0.011 | SVC |
| 0.591 +/- 0.029 | NB  |
| 0.632 +/- 0.011 | KNN |

### Under-sampling

```python
('scaler', StandardScaler()),
('sampler', ClusterCentroids()),
('clf', SVM()),   
```

clf with default parameters

| Score           | clf |
|-----------------|-----|
| 0.662 +/- 0.011 | SVC |
| 0.602 +/- 0.017 | NB  |
| 0.441 +/- 0.022 | KNN |