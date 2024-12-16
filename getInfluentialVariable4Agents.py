"""
===================================
Directed Information (DI) Algorithm
===================================
with binary Classifer

"""

# Author and Date: Yao Hu, 12/16/2024

print(__doc__)

import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')

import time
import math
import numpy as np
from statistics import variance

from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing

TIMEFORMAT = '%m/%d/%y %H:%M'
TIMEFORMAT1 = '%m/%d/%y'

# Data fitting
def datafitBRT(dataset, Y, par=None):
    indexNameArr = dataset.index.values
    labels = []

    out = []
    out.append(Y)
    if par:
        out.append(par)

    for rm in out:
        label = indexNameArr[rm]
        labels.append(label)

    xv = dataset.drop(labels).transpose()
    yv_obs = dataset.iloc[Y].transpose()

    classifer = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, algorithm='SAMME')

    classifer.fit(xv, yv_obs)

    # calculate the variance
    yv_pred = classifer.predict(xv)

    # calculate the variance
    err = abs(yv_obs - yv_pred)
    mse = math.sqrt(variance(err))

    return mse


# Estimate the value of directed information
def getDIgraph(newdatasets, Tmark):

    datasets = newdatasets

    m = datasets.shape[0]
    n = datasets.shape[1]
    Y = 0
    DI = {}

    for X in range(m):

        if X != Y:
            # get error variance for full model
            sigma_est_full = datafitBRT(datasets, Y, )

            # remove the columns indexed by Y and X, and get error variance for partial model
            sigma_est_part = datafitBRT(datasets, Y, X)

            temp_parent = X

            if sigma_est_full:
                # get the value of the directed information
                DI_value = 0.5 * abs(np.log(sigma_est_part/sigma_est_full))

            # MDL: minimum description length
            MDL = Tmark * 0.5 * math.log(n, 2)/n

            if DI_value < MDL:
                print('\t {0} : below the threshold {1:.4f}. \n'.format(temp_parent, MDL))
            else:
                # store the influential variables
                DI[str(temp_parent)] = DI_value

    # sort and print influential variables by DI_values
    varNames = datasets.index.values

    print('The influential variables of the target variable, {0} (larger than {1: .4f})'.format(varNames[Y], MDL))

    for key, value in sorted(DI.items(), key=lambda item: item[1], reverse=True):
        print("{0}, {1}: {2:.4f}\n".format(key, varNames[int(key)], value))


start = time.time()

#########################################################################################
# select the influential variables using directed information
#########################################################################################

print('Find the influential variables:')

# agents in different clusters
cluster1 = [10, 15, 16, 17, 21, 22, 23, 25, 26, 29]
cluster2 = [1, 5, 6, 12, 13, 14, 18, 20, 24, 27, 28, 30, 31, 32, 36, 37, 38, 39, 43, 44, 45, 46]
cluster3 = [2, 3, 4, 7, 8, 9, 11, 19, 40, 47, 48]

file_path = './agentdata/'

# Step 1: Select influential variables

datasets = pd.read_csv(file_path + 'agentdata_cluster1.csv')
# datasets = pd.read_csv(file_path + 'agentdata_cluster2.csv')
# datasets = pd.read_csv(file_path + 'agentdata_cluster3.csv')

# extract the values of the column, IrrDepth(inch)
irrDepth = datasets['Irrigation_Depth']

# convert irrDepth from real number to binary number if irrDepth > 0, then 1, otherwise 0
irrigEvent = pd.DataFrame(np.where(irrDepth > 0, 1, 0), columns=['IrrDepth_Depth'])

# replace irrDepth with irrigEvent
rdatasets = datasets

# change the column name, Irrigation_Depth to Irrigation_Event
rdatasets = rdatasets.rename(columns={'Irrigation_Depth': 'Irrigation_Event'})

# normalized rdatasets with l2 norm
tempdatasets = preprocessing.normalize(rdatasets, norm='l2')

# reshuffle the dataset
ntempdatasets = pd.DataFrame(tempdatasets, columns=rdatasets.columns.to_list())
ntempdatasets['Irrigation_Event'] = irrigEvent
newnormdatasets = ntempdatasets.sample(frac=1.0)

# transpose the dataset
newnormdatasets = newnormdatasets.transpose()

# Step 2: find and print the causal variables using Directed Information
# Tmark : length of markov chain
Tmark = 1

DI_opt = getDIgraph(newnormdatasets, Tmark)

end = time.time()
print("Time: {0:.4f}s".format(end - start))
