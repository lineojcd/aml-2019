import os
import sys
dir_cur = os.path.dirname(__file__)
dir_par = os.path.dirname(dir_cur)
sys.path.insert(0, dir_par)
from helper.tuning import tune
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
import sklearn
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV, ShuffleSplit

X = pd.read_csv("data/X_train.csv", index_col=0).values
y = pd.read_csv('data/y_train.csv', index_col=0).values
y = sklearn.utils.validation.column_or_1d(y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', ClusterCentroids()),
    ('clf', SVC()),
])

param_grid = {
}

param_rf = {
    'n_estimators': np.linspace(10, 100, num=4, dtype=np.int),
}

param_svm = {
    'C': [1.0, 2.0],
}

param_knn = {
    'n_neighbors': [1, 5],
}


def main():
    pipeline.fit(X, y)
    X_test = pd.read_csv('data/X_test.csv', index_col=0).values
    pred = pipeline.predict(X_test)
    np.savetxt('submission.csv', np.dstack((np.arange(0, pred.size), pred))[0], '%0.1f,%0.1f',
               comments='', header="id,y")
    print(np.unique(pred, return_counts=True))


if __name__ == "__main__":
    main()
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cv_score = cross_val_score(
        pipeline, X, y, cv=cv, scoring='balanced_accuracy')
    print("{:.3f} +/- {:.3f}".format(np.mean(cv_score), np.std(cv_score)))
    # param_grid = {'clf__'+k: v for k, v in param_svm.items()}
    # param_grid['smote'] = [SMOTE(), RandomOverSampler()]
    # tune(pipeline, param_grid, X, y, save=1, scoring='balanced_accuracy')
