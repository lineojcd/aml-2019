import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-f', '--file', action='store')
parser.add_argument('-n', '--name', action='store')

# # plot
# if graph:
#     plt.figure(figsize=(8, 8))
#     plt.errorbar(params, means, yerr=stds)

#     # plt.axhline(y=best_mean + best_stdev, color='red')
#     # plt.axhline(y=best_mean - best_stdev, color='red')
#     plt.plot(best_param, best_mean, 'or')

#     plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(grid_search.best_score_))
#     # plt.xlabel(param_name)
#     # plt.ylabel('Score')
#     plt.show()


args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.file)

    cv_score = df.set_index(['n_features', 'model']).cv_score
    train_score = df.set_index(['n_features', 'model']).train_score
    a= cv_score.unstack().plot(kind='bar', stacked=False)
    print(type(a), dir(a))
    # plt.savefig('plot/cv_score_{}.png'.format(args.name))
    train_score.unstack().plot(kind='bar', stacked=False)
    plt.savefig('plot/train_score_{}.png'.format(args.name))
