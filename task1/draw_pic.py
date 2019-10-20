import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-f', '--file', action='store')
parser.add_argument('-n', '--name', action='store')

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.file)

    cv_score = df.set_index(['n_features', 'model']).cv_score
    train_score = df.set_index(['n_features', 'model']).train_score
    cv_score.unstack().plot(kind='bar', stacked=False)
    plt.savefig('plot/cv_score_{}.png'.format(args.name))
    train_score.unstack().plot(kind='bar', stacked=False)
    plt.savefig('plot/train_score_{}.png'.format(args.name))
