import os
from time import time, strftime
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from matplotlib import pyplot as plt
from .visual import grid_search_plot, plot_grid_search_all


def tune(pipeline, param_grid, X, y, save=True, scoring='r2'):
    if save and not os.path.exists('log'):
        os.makedirs('log')
    timestr = strftime("%Y%m%d-%H%M%S")
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv,
                               n_jobs=-1)
    t0 = time()
    grid_search.fit(X, y)

    print("done in {:.3f}s\n".format(time() - t0))
    print("Best score: {:.3f} +/- {:.3f}".format(grid_search.best_score_,
                                                 grid_search.cv_results_['std_test_score'][grid_search.best_index_]))
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    if save:
        plot_grid_search_all(
            grid_search, fname='log/gscv_{}.png'.format(timestr))
        joblib.dump(grid_search, 'log/gscv_{}.pkl'.format(timestr), compress=1)
        joblib.dump(grid_search.best_estimator_,
                    'log/best_{}.pkl'.format(timestr), compress=1)
