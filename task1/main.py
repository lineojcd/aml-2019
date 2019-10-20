import os
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn.impute
import sklearn.ensemble
import sklearn.neighbors
import sklearn.feature_selection
import csv

# supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Model(object):

    def __init__(self, X: np.ndarray, y: np.ndarray, n_features: int, model_name: str = 'rf'):
        self.X = X
        self.y = y
        self.n_features = n_features
        self.model_name = model_name


        # keep a list of object wiht transform() for test data
        self.transformation = []
        self.process_all()

    def _transform_data(self, transformer):
        "Apply the same transformation to training data and testing data"
        self.X = transformer.transform(self.X)
        self.transformation.append(transformer)

    def data_impute(self, strategy: str = 'mean', verbose=True):
        if verbose:
            print('{:<10d} missing data replaced using {}'.format(np.count_nonzero(np.isnan(self.X)), strategy))
        imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy=strategy, verbose=1).fit(self.X)
        scaler = preprocessing.StandardScaler().fit(self.X)
        
        self._transform_data(imp)
        self._transform_data(scaler)


    def process_outliers(self, method: str = 'IsolationForest', verbose=True):
        if method == 'IsolationForest':
            outliers = sklearn.ensemble.IsolationForest(
                n_estimators=300, max_samples=1000).fit_predict(self.X, self.y)
        elif method == 'LocalOutlierFactor':
            outliers = sklearn.neighbors.LocalOutlierFactor(
                contamination="auto").fit_predict(self.X, self.y)

        self.X = self.X[outliers == 1]
        self.y = self.y[outliers == 1]
        if verbose:
            print('{:<10d} outliers detected by {}'.format(len(np.where(outliers == 0)), method))

    def feature_transformation(self, method: str = 'pca', verbose=True):
        if method == 'pca':
            feature_transformer = sklearn.decomposition.PCA(n_components=self.n_features).fit(self.X)
        else:
            raise ValueError('{} not included in class Model'.format(method))
        self._transform_data(feature_transformer)
        if verbose:
            print('{:<10} features after transformation using {}'.format(
                self.X.shape[1], method))

    def process_all(self, strategy='mean', method='LocalOutlierFactor',
                    feature_transformation='pca', verbose=1):
        self.data_impute(strategy, verbose)
        self.process_outliers(method, verbose)
        self.feature_transformation(feature_transformation, verbose)

    def train(self):
        self.model.fit(self.X, self.y)
        return self.model.score(self.X, self.y)

    def predict(self, X_test):
        X_test_ = X_test
        for i in self.transformation:
            X_test_ = i.transform(X_test_)
        return self.model.predict(X_test_)

    def cv(self):
        return np.mean(cross_val_score(self.model, self.X, self.y, cv=5))

    def choose_model(self, model: str):
        self.model_name = model
        if model == 'ridge':
            self.model = self._ridge()
        elif model == 'rf':
            self.model = self._rf()
        elif model == 'svr_rbf':
            self.model = self._svr_rbf()
        elif model == 'svr_rbf':
            self.model = self._svr_lin()
        elif model == 'svr_poly':
            self.model = self._svr_poly()
        else:
            raise ValueError('{} not included in class Model'.format(model))

    def _ridge(self):
        return sklearn.linear_model.Ridge(alpha=1)

    def _svr_rbf(self):
        "Default kernel: rbf"
        return sklearn.svm.SVR()

    def _svr_lin(self):
        "Default kernel: lin"
        return sklearn.svm.SVR(kernel='linear')

    def _svr_poly(self):
        "Default kernel: poly"
        return sklearn.svm.SVR(kernel='poly')

    def _rf(self):
        return sklearn.ensemble.RandomForestRegressor(n_estimators=500, max_depth=15, n_jobs=-1)


def main():
    X = pd.read_csv('data/X_train.csv', index_col=0).values
    y = pd.read_csv('data/y_train.csv', index_col=0).values
    y = sklearn.utils.validation.column_or_1d(y)

    print('{:<10} training samples'.format(X.shape[0]))
    print('{:<10} features before transformation\n'.format(X.shape[1]))

    # n_features = [10, 12, 15, 17, 20, 50]
    # model_list = ['ridge', 'svr_rbf', 'rf']

    # score_list = []

    # for m in model_list:
    #     for n in n_features:
    #         model = Model(X, y, n)
    #         model.choose_model(m)
    #         train_score = model.train()
    #         cv_score = model.cv()
    #         print()
    #         print('{:<10}'.format(model.model_name))
    #         print('{:<10f} training'.format(train_score))
    #         print('{:<10f} CV'.format(cv_score))
    #         print()

    #         score_list.append('{},{:.8f},{:.8f},{}\n'.format(model.model_name, train_score,
    #                                                cv_score, n))

    # with open('compare_pca.csv', 'w') as writeFile:
    #     writeFile.write('model,train_score,cv_score,n_features\n')
    #     for line in score_list:
    #         writeFile.write(line)

    model = Model(X, y, 10)
    model.choose_model('rf')

    print()
    print('{:<10}'.format(model.model_name))
    print('{:<10f} training'.format(model.train()))
    print('{:<10f} CV'.format(model.cv()))
    print()    

    X_test = pd.read_csv('data/X_test.csv', index_col=0).values
    pred = model.predict(X_test)
    np.savetxt(".data/submission.csv", np.dstack((np.arange(0, pred.size), pred))[0], "%0.1f,%f",
               comments='', header="id,y")


if __name__ == '__main__':
    main()
