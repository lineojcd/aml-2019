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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV, ShuffleSplit
from lightgbm import LGBMClassifier

X = pd.read_csv("{}/data/X_train.csv".format(dir_cur), index_col=0).values
y = pd.read_csv("{}/data/y_train.csv".format(dir_cur), index_col=0).values
y = sklearn.utils.validation.column_or_1d(y)

def outliers_removal(X, y):
    outliers = LocalOutlierFactor().fit_predict(X, y)
    return X[outliers == 1], y[outliers == 1]

X, y = outliers_removal(X, y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', ClusterCentroids()),
    ('clf', RandomForestClassifier(random_state=1)),
])

param_grid = {
    'clf__n_estimators': [50, 100, 200, 500],
}

def main():
    pipeline.fit(X, y)
    X_test = pd.read_csv("{}/data/X_test.csv".format(dir_cur), index_col=0).values
    pred = pipeline.predict(X_test)
    np.savetxt("{}/submission.csv".format(dir_cur), np.dstack((np.arange(0, pred.size), pred))[0], '%0.1f,%0.1f',
               comments='', header="id,y")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print('X shape: {},\t y shape: {}'.format(X.shape, y.shape))
    # main()
    # param_grid = {'clf__'+k: v for k, v in param_svm.items()}
    tune(pipeline, param_grid, X, y, save=1, scoring='balanced_accuracy', verbose=0)
