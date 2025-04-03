"""
=========================================
Predict irrigation depth for each cluster
=========================================
"""
# Author: Yao Hu
# Date: 12/16/2024
# Update: 3/11/2025

print(__doc__)

import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings("ignore")

# predict monthly irrigation depth for nonstationary agents in each cluster using XGBoost
# monthly data for each agent in each cluster is used as the input
# data is stored in files with the format of 'agentdata_cluster1.csv', 'agentdata_cluster2.csv' and
# 'agentdata_cluster3.csv' in the folder, 'agentdata'.
# each file contains 8 columns: 'Irrigation_Depth', 'Corn', 'Wheat', 'Soybeans', 'Sorghum', 'Diesel', 'Precipitation',
# 'Temperature' and each row is the monthly data for an agent in each cluster.
# the output is the monthly irrigation depth for specific agents in each cluster.

def trainXGBoost1(X_train, y_train, X_test, y_test):
    # fit model no training data
    irr_reg_mod = xgb.XGBRegressor(
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=5,
        subsample=0.5,
        n_estimators=1000,
        objective='reg:squarederror',
        colsample_bytree=1,
        reg_alpha=0.75,
        reg_lambda=0.45,
        gamma=0,
        seed=42)

    irr_reg_mod.fit(X_train, y_train)
    tr_irr_pred = irr_reg_mod.predict(X_train)

    # evaluate predictions
    tr_m_mse = mean_squared_error(y_train, tr_irr_pred)
    tr_m_r2 = r2_score(y_train, tr_irr_pred)
    print("MSE and R2 of Training: {0:.4f} and {1:.2f}".format(tr_m_mse, tr_m_r2))

    # make predictions for test data
    tt_irr_pred = irr_reg_mod.predict(X_test)
    # evaluate predictions
    tt_m_mse = mean_squared_error(y_test, tt_irr_pred)
    tt_m_r2 = r2_score(y_train, tr_irr_pred)
    print("MSE and R2 of Training: {0:.4f} and {1:.2f}".format(tt_m_mse, tt_m_r2))

    return irr_reg_mod


# agent list
agent_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
              22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48]

# agents in different clusters
cluster1 = [10, 15, 16, 17, 21, 22, 23, 25, 26, 29]
cluster2 = [1, 5, 6, 12, 13, 14, 18, 20, 24, 27, 28, 30, 31, 32, 36, 37, 38, 39, 43, 44, 45, 46]
cluster3 = [2, 3, 4, 7, 8, 9, 11, 19, 40, 47, 48]

# nonstationary agents
nonstationary_agent = [21, 22, 23, 31, 36, 37, 47, 48]

# # Step 1: train the XBGoost model for each cluster
# print('Train the XGBoost model for each cluster:')
# # load the data
# file_path = './data/'
# datasets = pd.read_csv(file_path + 'agentdata_cluster3.csv')
#
# # drop the rows with missing values
# datasets = datasets.dropna()
# # drop the rows with negative values
# datasets = datasets[(datasets['Irrigation_Depth'] >= 0) & (datasets['Corn'] >= 0) & (datasets['Wheat'] >= 0) &
#                     (datasets['Soybeans'] >= 0) & (datasets['Sorghum'] >= 0) & (datasets['Diesel'] >= 0) &
#                     (datasets['Precipitation'] >= 0)]
#
# # target variable: Irrigation_Depth
# # explanatory variables for cluster 1: Wheat, Precipitation, Temperature
# # explanatory variables for cluster 2: Wheat, Soybeans, Precipitation, Temperature
# # explanatory variables for cluster 3: Wheat, Soybeans, Precipitation, Temperature
#
# # split the data into training and testing sets
# # cluster 1
# # X_train1, X_test1, y_train1, y_test1 = train_test_split(datasets[['Wheat', 'Precipitation', 'Temperature']],
# #                                                         datasets['Irrigation_Depth'], test_size=0.3, random_state=0)
# # cluster 2
# # X_train2, X_test2, y_train2, y_test2 = train_test_split(datasets[['Wheat', 'Soybeans', 'Precipitation', 'Temperature']],
# #                                                         datasets['Irrigation_Depth'], test_size=0.3, random_state=0)
#
# # # cluster 3
# X_train3, X_test3, y_train3, y_test3 = train_test_split(datasets[['Wheat', 'Soybeans', 'Precipitation', 'Temperature']],
#                                                         datasets['Irrigation_Depth'], test_size=0.3, random_state=0)
#
# # train the XGBoost model for each cluster using a function
# # cluster 1, cluster 2 and cluster 3
# irr_reg_mod3 = trainXGBoost1(X_train3, y_train3, X_test3, y_test3)
# # save the model to disk
# filename = 'irr_reg_mod3.sav'
# pickle.dump(irr_reg_mod3, open(filename, 'wb'))


# Step 2: predict the irrigation depth for nonstationary agents in each cluster by Model 1 and Model 2
# load the models for different clusters from disk
filename = 'irr_reg_mod11.sav'
# load the model from disk using pickle
irr_reg_mod11 = pickle.load(open(filename, 'rb'))

filename = 'irr_reg_mod12.sav'
irr_reg_mod12 = pickle.load(open(filename, 'rb'))

filename = 'irr_reg_mod13.sav'
irr_reg_mod13 = pickle.load(open(filename, 'rb'))

filename = 'irr_reg_mod21.sav'
# load the model from disk using pickle
irr_reg_mod21 = pickle.load(open(filename, 'rb'))

filename = 'irr_reg_mod22.sav'
# load the model from disk using pickle
irr_reg_mod22 = pickle.load(open(filename, 'rb'))

filename = 'irr_reg_mod23.sav'
irr_reg_mod23 = pickle.load(open(filename, 'rb'))

# target variable: Irrigation_Depth
# explanatory variables for cluster 1: Wheat, Precipitation, Temperature
# explanatory variables for cluster 2: Wheat, Soybeans, Precipitation, Temperature
# explanatory variables for cluster 3: Wheat, Soybeans, Precipitation, Temperature

# load the data for agents in both nonstationary_agent and cluster 1
file_path = './data/'

# 2. Specify the feature columns
# Cluster 1
feature_cols = ['Wheat', 'Precipitation', 'Temperature']

# Clusters 2 and 3
# feature_cols = ['Wheat', 'Soybeans', 'Precipitation', 'Temperature']

for nonstat_agent in nonstationary_agent:
    # load the data in csv file
    datasets = pd.read_csv(file_path + 'agentdata_' + str(nonstat_agent) + '.csv')
    # drop the rows with missing values
    datasets = datasets.dropna()
    # drop the rows with negative values
    datasets = datasets[(datasets['Irrigation_Depth'] >= 0) & (datasets['Corn'] >= 0) & (datasets['Wheat'] >= 0) &
                        (datasets['Soybeans'] >= 0) & (datasets['Sorghum'] >= 0) & (datasets['Diesel'] >= 0) &
                        (datasets['Precipitation'] >= 0)]

    test1_mask = (datasets['Year'] > 2012)
    test2_mask = (datasets['Year'] > 2013)

    if nonstat_agent in cluster1:
        # predict the irrigation depth for nonstationary agents in cluster 1
        # Split into training features (X) and target (y)
        feature_cols = ['Wheat', 'Precipitation', 'Temperature']

        X1_test = datasets.loc[test1_mask, feature_cols]
        y1_test = datasets.loc[test1_mask, 'Irrigation_Depth']

        X2_test = datasets.loc[test2_mask, feature_cols]
        y2_test = datasets.loc[test2_mask, 'Irrigation_Depth']

        y_pred1 = irr_reg_mod11.predict(X1_test)
        y_pred2 = irr_reg_mod21.predict(X2_test)

    elif nonstat_agent in cluster2:
        # predict the irrigation depth for nonstationary agents in cluster 2
        feature_cols = ['Wheat', 'Soybeans', 'Precipitation', 'Temperature']

        X1_test = datasets.loc[test1_mask, feature_cols]
        y1_test = datasets.loc[test1_mask, 'Irrigation_Depth']

        X2_test = datasets.loc[test2_mask, feature_cols]
        y2_test = datasets.loc[test2_mask, 'Irrigation_Depth']

        y_pred1 = irr_reg_mod12.predict(X1_test)
        y_pred2 = irr_reg_mod22.predict(X2_test)

    else:
        # predict the irrigation depth for nonstationary agents in cluster 3
        feature_cols = ['Wheat', 'Soybeans', 'Precipitation', 'Temperature']

        X1_test = datasets.loc[test1_mask, feature_cols]
        y1_test = datasets.loc[test1_mask, 'Irrigation_Depth']

        X2_test = datasets.loc[test2_mask, feature_cols]
        y2_test = datasets.loc[test2_mask, 'Irrigation_Depth']

        y_pred1 = irr_reg_mod13.predict(X1_test)
        y_pred2 = irr_reg_mod23.predict(X2_test)

    # check the predicted irrigation depth and if it is negative, set it to zero
    y_pred1 = np.where(y_pred1 < 1, 0, y_pred1)
    y_pred2 = np.where(y_pred2 < 1, 0, y_pred2)

    # evaluate predictions
    mse3 = mean_squared_error(y1_test, y_pred1)
    r23 = r2_score(y1_test, y_pred1)
    # print the results for each agent
    print("Model 1: Agent {0}: MSE and R2: {1:.4f} and {2:.2f}".format(nonstat_agent, mse3, r23))

    mse3 = mean_squared_error(y2_test, y_pred2)
    r23 = r2_score(y2_test, y_pred2)
    print("Model 2: Agent {0}: MSE and R2: {1:.4f} and {2:.2f}".format(nonstat_agent, mse3, r23))

    # add the predicted irrigation depth to the dataframe
    datasets['Irrigation_Depth_M1'] = datasets['Irrigation_Depth']
    datasets['Irrigation_Depth_M2'] = datasets['Irrigation_Depth']

    # Define Boolean masks to pick out rows from the relevant years
    mask_2012 = (datasets['Year'] > 2012)
    mask_2013 = (datasets['Year'] > 2013)

    # Overwrite the relevant rows with the prediction arrays
    datasets.loc[mask_2012, 'Irrigation_Depth_M1'] = y_pred1
    datasets.loc[mask_2013, 'Irrigation_Depth_M2'] = y_pred2

    # Save the predicted irrigation depth to csv file
    datasets.to_csv(file_path + 'agentdata_' + str(nonstat_agent) + '_pred_m1_m2.csv', index=False)
