import os
import sys
dir_cur = os.path.abspath(os.path.dirname(__file__))
dir_par = os.path.dirname(dir_cur)
sys.path.insert(0, dir_par)
from helper.tuning import tune
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sklearn
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Reshape, Flatten, GlobalAveragePooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline

from main import read_data

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(200, input_dim=1000, activation='relu'))
	model.add(Dense(40, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
	return model

nn = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=1)

pipeline = Pipeline([
	('scaler', StandardScaler()),
	# ('under_sampler', ClusterCentroids()),
	('combined_sampler', SMOTEENN()), # worse than under sampling alone
	('nn', nn),
])

if __name__ == "__main__":
	X, y = read_data()
	cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
	cv_score = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy')
	print("Score: {:.3f} +/- {:.3f}".format(np.mean(cv_score), np.std(cv_score)))