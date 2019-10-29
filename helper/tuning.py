import os
from time import time, strftime
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from matplotlib import pyplot as plt
import pprint
from .visual import grid_search_plot, plot_grid_search_all


def tune(pipeline, param_grid, X, y, save=True, scoring='r2', verbose=0):
    if save and not os.path.exists('log'):
        os.makedirs('log')
    timestr = strftime("%Y%m%d-%H%M%S")
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv,
                               n_jobs=-1, verbose=verbose)
    grid_search.fit(X, y)

    for mean, std, param in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score'], grid_search.cv_results_['params']):
        print("Score: {:.3f} +/- {:.3f}".format(mean, std))
        pprint.pprint(param)

    if save:
        plot_grid_search_all(
            grid_search, fname='log/gscv_{}.png'.format(timestr))
        joblib.dump(grid_search, 'log/gscv_{}.pkl'.format(timestr), compress=1)
        joblib.dump(grid_search.best_estimator_,
                    'log/best_{}.pkl'.format(timestr), compress=1)
