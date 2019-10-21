from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import pandas as pd
import pprint
import math

def grid_search_plot(grid_search, param_name, ax=None, fit_time=False):
    assert (isinstance(grid_search, pd.DataFrame)
            or isinstance(grid_search, GridSearchCV)
            ), 'grid_search must be of GridSearchCV or pandas.DataFrame'
    if ax is None:
        ax = plt.gca()
    if not isinstance(grid_search, pd.DataFrame):
        df = pd.DataFrame(grid_search.cv_results_)

    df.sort_values(by='rank_test_score')
    best_row = df.iloc[0, :]
    best_mean = best_row['mean_test_score']
    best_param = best_row['param_' + param_name]

    if fit_time:
        to_plot = 'fit_time'
        ax.set_ylabel('fit_time')
    else:
        to_plot = 'test_score'
        # plot the best parameters
        ax.plot(best_param, best_mean, 'or')
        ax.set_ylabel('Score')

    means = df['mean_{}'.format(to_plot)]
    stds = df['std_{}'.format(to_plot)]
    params = df['param_' + param_name]

    ax.errorbar(params, means, yerr=stds)
    ax.set_xlabel(param_name)

    return ax


def plot_grid_search_all(grid_search, fname=None, fit_time=False, title=None):

    param_all = list(grid_search.param_grid.keys())
    n = len(param_all)
    if n < 4:
        fig, axes = plt.subplots(
            nrows=1, ncols=len(param_all), figsize=(3*n, 3))

        for i in range(n):
            grid_search_plot(
                grid_search, param_all[i], axes[i], fit_time=fit_time)
        plt.tight_layout()
    else:
        nrows = math.ceil(math.sqrt(n))
        fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(3*n, 3*n), fit_time=fit_time)
        for i in range(n):
            grid_search_plot(grid_search, param_all[i], axes[i])
        plt.tight_layout()

    if not title:
        title = '{} vs param'.format(
            'Score') if not fit_time else '{} vs param'.format('fit_time')
    st = fig.suptitle(title, fontsize="x-large")
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    if fname:
        plt.savefig(fname, dpi=200)

    plt.close()
    return fig
