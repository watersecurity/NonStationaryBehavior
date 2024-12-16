# Perform Augmented Dickey-Fuller test to evaluate the non-stationarity of agents’ irrigation behavior
# Author and Date: Yao Hu, 12/06/2024

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(y):
    """
    Perform Augmented Dickey-Fuller test

    Parameters
    ----------
    y : array_like
        The time series data.

    Returns
    -------
    result : tuple
        A tuple containing the test statistic, p-value, optimal lag length, and critical values.
    """

    # Null Hypothesis (H0): The time series has a unit root (i.e., it is non-stationary).
    # Alternate Hypothesis (H1): The time series does not have a unit root (i.e., it is stationary).
    # Markov assumption: maxlag = 0
    result = adfuller(y, maxlag=0, autolag=None, regression='ct')
    return result[0], result[1], result[2], result[4]


def is_non_stationary(y, significance_level=0.05):

    test_statistic, p_value, optimal_lag, critical_values = adf_test(y)

    print(f"Test Statistic: {test_statistic}")
    print(f"P-value: {p_value}")
    print(f"Optimal Lag: {optimal_lag}")
    print("Critical Values:")

    for key, value in critical_values.items():
        print(f"  {key}: {value}")

    # If the p-value of the Augmented Dickey-Fuller (ADF) test is smaller than the significance level (e.g., 0.05),
    # you reject the null hypothesis.
    if p_value < significance_level:
        print("Conclusion: The time series is stationary.")
        return 0
    else:
        print("Conclusion: The time series is non-stationary.")
        return 1

if __name__ == "__main__":

    # Load the cluster_agent_precip_ratio.csv dataset
    df = pd.read_csv("cluster_agent_gw_precip_ratio.csv")

    # Loop through the rows of the dataframe and check for non-stationarity
    for index, row in df.iterrows():
        print(f"Row {index}:")
        gw2precip_cluster = row.values
        # Remove the first and second elements of gw2precip
        gw2precip = gw2precip_cluster[2:]
        # Check for non-stationarity
        stationary_status = is_non_stationary(gw2precip)
        # Add the stationary status to the dataframe
        df.at[index, "stationary_status"] = stationary_status

    # Save the dataframe to a csv file
    df.to_csv("new_cluster_agent_gw_precip_ratio_stationary.csv", index=False)
