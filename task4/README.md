# task1 

Sleep Staging

## Problem description

In this task we will perform sequence classification. We will categorize temporally coherent and uniformly distributed short sections of a long time-series. In particular, for each 4 seconds of a lengthy EEG/EMG measurement of brain activity recorded during sleep, we will assign one of the 3 classes corresponding to the sleep stage present within the evaluated epoch. 

## Data

3 channel of the data were concatenate, forming a `X_train` of shape `(64800, 1536)`. The signals were measured in 3 channels at the same time with the same measure frequency.
