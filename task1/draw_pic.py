import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('compare_pca.csv')

cv_score = df.set_index(['n_features', 'model']).cv_score
train_score = df.set_index(['n_features', 'model']).train_score
cv_score.unstack().plot(kind='bar', stacked=False)
plt.savefig('cv_score_pca.png')
train_score.unstack().plot(kind='bar', stacked=False)
plt.savefig('train_score_pca.png')
