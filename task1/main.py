import warnings
import csv
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from plot_grid_search import plot_grid_search_all
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from sklearn.externals import joblib


class LocalOutlierTransformer(TransformerMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, X):
        lcf = LocalOutlierFactor(**self.kwargs)
        outliers = lcf.fit_predict(X)
        return X[outliers == 1]

    def fit(self, *args, **kwargs):
        return self

X = pd.read_csv("data/X_train.csv", index_col=0).values
y = pd.read_csv('data/y_train.csv', index_col=0).values
y = sklearn.utils.validation.column_or_1d(y)
X_test = pd.read_csv('data/X_test.csv', index_col=0).values

data_pre = Pipeline([
    ('sim', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

feature_kbest = Pipeline([
    ('var', VarianceThreshold(0.05)),
    ('kbest', SelectKBest(f_regression, k=200))
])

pipeline = Pipeline([
    ('data_pre', data_pre),
    ('feature_extract', feature_kbest),
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=500, max_depth=15, n_jobs=-1)),
])

def main():
    pipeline.fit(X, y)
    pred = pipeline.predict(X_test)
    np.savetxt('submission.csv', np.dstack((np.arange(0, pred.size), pred))[0], '%0.1f,%f',
               comments='', header="id,y")

def tune(pipeline, param_grid, X, y, save=True):
    from time import time, strftime
    timestr = strftime("%Y%m%d-%H%M%S")
    grid_search = GridSearchCV(pipeline, param_grid, scoring='r2', cv=5)
    t0 = time()
    grid_search.fit(X, y)
    
    print("done in %0.3fs\n" % (time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    if save:
        if not os.path.exists('log'):
            os.makedirs('log')
        plot_grid_search_all(grid_search, fname='log/gscv_{}.png'.format(timestr))
        joblib.dump(grid_search, 'log/gscv_{}.pkl'.format(timestr), compress=1)
        joblib.dump(grid_search.best_estimator_, 'log/best_{}.pkl'.format(timestr), compress=1)

param_grid = {
    'data_pre__sim__strategy': ['mean', 'median'],
    'feature_extract__var__threshold': [0.01, 0.05],
    'feature_extract__kbest__k': np.linspace(5, 400, num=5, dtype=np.int),
}

if __name__ == '__main__':
    main()
    # tune(pipeline, param_grid, X, y)