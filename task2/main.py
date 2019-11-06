import os
import sys
dir_cur = os.path.abspath(os.path.dirname(__file__))
dir_par = os.path.dirname(dir_cur)
sys.path.insert(0, dir_par)
from helper.tuning import tune
import warnings
warnings.filterwarnings("ignore")

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
from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline


def read_data():
    X = pd.read_csv("{}/data/X_train.csv".format(dir_cur), index_col=0).values
    y = pd.read_csv("{}/data/y_train.csv".format(dir_cur), index_col=0).values
    y = sklearn.utils.validation.column_or_1d(y)
    print('Reading data...')
    print('X: {}\ny: {}\n'.format(X.shape, y.shape))
    return X, y


def outliers_removal(X, y):
    outliers = LocalOutlierFactor().fit_predict(X, y)
    return X[outliers == 1], y[outliers == 1]


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # ('var', VarianceThreshold(0.01)),
    # ('kbest', SelectKBest(f_regression, k=700)),
    ('cc', ClusterCentroids()),
    ('clf', SVC(gamma='scale', class_weight='balanced', decision_function_shape='ovo',
                kernel='rbf')),
])

param_grid = {
    # 'clf__n_estimators': [50, 100, 200, 500],
    'kbest__k': [200, 300, 400, 500, 600, 700, 800, 900, 1000]
}


def main():
    pipeline.fit(X, y)
    X_test = pd.read_csv(
        "{}/data/X_test.csv".format(dir_cur), index_col=0).values
    pred = pipeline.predict(X_test)
    np.savetxt("{}/submission.csv".format(dir_cur),
               np.dstack((np.arange(0, pred.size), pred))[0],
               '%0.1f,%0.1f', comments='', header="id,y")


if __name__ == "__main__":
    import time
    t0 = time.time()
    X, y = read_data()
    # X, y = outliers_removal(X, y)
    # main()
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    cv_score = cross_val_score(
        pipeline, X, y, scoring='balanced_accuracy', cv=cv)
    # pipeline.set_params(clf__gamma=gamma)
    print('Time : {:.3f} sec'.format(time.time() - t0))
    print("Score: {:.3f} +/- {:.3f}".format(np.mean(cv_score), np.std(cv_score)))
    # param_grid = {'clf__'+k: v for k, v in param_svm.items()}
    # tune(pipeline, param_grid, X, y, save=1, scoring='balanced_accuracy', verbose=1)
