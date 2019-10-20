

def GridSearch_table_plot(grid_search, param_name,
                          graph=True,
                          display_all_params=False,
                          save=None):
    from matplotlib import pyplot as plt
    import pandas as pd
    from sklearn.externals import joblib

    if save:
        joblib.dump(grid_search.best_params_, '{}.pkl'.format(
            save), compress=1)  # Only best parameters

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    best_mean = best_row['mean_test_score']
    best_param = best_row['param_' + param_name]

    # =====================================
    if display_all_params:
        print(
            "best score:      {:0.5f} (+/-{:0.5f})".format(grid_search.best_score_, grid_search.cv_results_['std_test_score'][grid_search.best_index_]))
        import pprint
        pprint.pprint(grid_search.best_estimator_.get_params())

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        # plt.axhline(y=best_mean + best_stdev, color='red')
        # plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(grid_search.best_score_))
        # plt.xlabel(param_name)
        # plt.ylabel('Score')
        plt.show()

    return scores_df
