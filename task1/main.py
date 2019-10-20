import os
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
import sklearn.impute
import sklearn.ensemble
import sklearn.neighbors
import sklearn.feature_selection
import csv
from lightgbm import LGBMRegressor
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

class Model(object):

    def __init__(self, X: np.ndarray, y: np.ndarray, n_features: int, model_name: str = 'rf', verbose=True):
        self.X = X
        self.y = y
        self.n_features = n_features
        self.model_name = model_name

        # keep a list of object with transform() for test data
        self.transformation = []
        self.process_all(verbose=verbose)

    def _transform_data(self, transformer):
        "Apply the same transformation to training data and testing data"
        self.X = transformer.transform(self.X)
        self.transformation.append(transformer)

    def data_impute(self, strategy='mean'):
        imp = sklearn.impute.SimpleImputer(
            missing_values=np.nan, strategy=strategy, verbose=1).fit(self.X)
        scaler = preprocessing.StandardScaler().fit(self.X)

        self._transform_data(imp)
        self._transform_data(scaler)

    def process_outliers(self, method='IsolationForest', verbose=True):
        if method == 'IsolationForest':
            outliers = sklearn.ensemble.IsolationForest(
                n_estimators=300, max_samples=1000).fit_predict(self.X, self.y)
        elif method == 'LocalOutlierFactor':
            outliers = sklearn.neighbors.LocalOutlierFactor(
                contamination="auto").fit_predict(self.X, self.y)

        self.X = self.X[outliers == 1]
        self.y = self.y[outliers == 1]

    def feature_transformation(self, method: str = 'pca'):
        if method == 'pca':
            feature_transformer = sklearn.decomposition.PCA(
                n_components=self.n_features).fit(self.X)
        elif method == 'varience':
            var = sklearn.feature_selection.VarianceThreshold(0.01).fit(self.X)
            self._transform_data(var)
            feature_transformer = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_regression,
                                                        k=self.n_features).fit(self.X, self.y)
        else:
            raise ValueError('{} not included in class Model'.format(method))

        self._transform_data(feature_transformer)

    def process_all(self, strategy='mean', method='LocalOutlierFactor',
                    feature_transformation='varience', verbose=False):

        n_nan = np.count_nonzero(np.isnan(self.X))
        (n_ori, n_fea) = self.X.shape
        self.data_impute(strategy)
        self.process_outliers(method)
        self.feature_transformation(feature_transformation)
        if verbose:
            print('{:<10d} missing data in {} replaced using {}'.format(
                n_nan, n_ori, strategy))
            print('{:<10d} outliers detected by {}'.format(
                n_ori - self.X.shape[0], method))
            print('{:<10d} features from {} after transformation using {}'.format(
                self.X.shape[1], n_fea, feature_transformation))

    def train(self):
        self.model.fit(self.X, self.y)
        return self.model.score(self.X, self.y)

    def predict(self, X_test):
        X_test_ = X_test
        for i in self.transformation:
            X_test_ = i.transform(X_test_)
        return self.model.predict(X_test_)

    def cv(self):
        score = np.mean(cross_val_score(self.model, self.X, self.y, cv=5))
        return score

    def choose_model(self, model: str):
        self.model_name = model
        if model == 'ridge':
            self.model = self._ridge()
        elif model == 'rf':
            self.model = self._rf()
        elif model == 'svr':
            self.model = self._svr()
        elif model == 'lgbm':
            self.model = self._lgbm()
        else:
            raise ValueError('{} Not included'.format(model))

    def tune_model(self, model: str, param, iter=100, cv=5):
        self.choose_model(model)
        rscv = RandomizedSearchCV(estimator=self.model, param_distributions=param,
                                  n_iter=iter, scoring='r2', cv=cv, verbose=False, n_jobs=-1)
        rscv.fit(self.X, self.y, verbose=False)

        print("{:<10f}: CV = {}, n_iter={}".format(rscv.best_score_, cv, iter),
              "obtained for\n", rscv.best_params_, '\n')

    def _lgbm(self):
        return LGBMRegressor(max_depth=-1, metric='None')

    def _ridge(self):
        return sklearn.linear_model.Ridge(alpha=1)

    def _svr(self):
        "Default kernel: rbf"
        return sklearn.svm.SVR()

    def _rf(self):
        return sklearn.ensemble.RandomForestRegressor(n_estimators=500, max_depth=15, n_jobs=-1)


def main():
    X = pd.read_csv("data/X_train.csv", index_col=0).values
    y = pd.read_csv('data/y_train.csv', index_col=0).values
    y = sklearn.utils.validation.column_or_1d(y)

    # n_features = np.linspace(50, 300, num=6, dtype=np.int)
    # model_list = ['rf']
    # score_list = []

    # for n in n_features:
    #     for m in model_list:
    #         model = Model(X, y, n, verbose=True)
    #         model.choose_model(m)
    #         score_list.append('{},{:.8f},{:.8f},{}\n'.format(model.model_name, model.train(),
    #                                                         model.cv(), model.X.shape[1]))

    # with open('compare_varience.csv', 'w') as writeFile:
    #     writeFile.write('model,train_score,cv_score,n_features\n')
    #     for line in score_list:
    #         writeFile.write(line)

    model = Model(X, y, 225, verbose=True)
    model.choose_model('rf')
    model.train()
    X_test = pd.read_csv('data/X_test.csv', index_col=0).values
    pred = model.predict(X_test)
    np.savetxt('submission.csv', np.dstack((np.arange(0, pred.size), pred))[0], '%0.1f,%f',
               comments='', header="id,y")


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    main()
