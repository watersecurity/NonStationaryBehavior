# Conduct counterfactural analysis of the groundwater behavior
# in terms of groundwater irrigation depth (mm) and relative groundwater level (mm).
# Author: Yao Hu
# Date: 12/16/2024
# Update: 03/23/2025

import pandas as pd
import numpy as np
from matplotlib import style, pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# non-stationary agent
# nonstationary_agent = [21, 22, 23, 31, 36, 37, 47, 48]

df_gwu = pd.read_csv('./data/nonstat/agents_wu_48.csv', header=None)
df_gwu.columns = ['Year', 'Actual GWU', 'Predicted GWU (M1)', 'Predicted GWU (M2)']
# Assume df_gwu has columns: ['Year', 'Actual GWU', 'Predicted GWU (M1)', 'Predicted GWU (M2)']
mask_m1_gwu = (df_gwu['Year'] >= 2012) & (df_gwu['Year'] <= 2022)
mask_m2_gwu = (df_gwu['Year'] >= 2013) & (df_gwu['Year'] <= 2022)

# --------------------
# 2. PREPARE THE GWL DATA & DIFFERENCES
# --------------------
# Assume df_gwl has columns: ['Year', 'Actual GWL', 'Predicted GWL (M1)', 'Predicted GWL (M2)']
# Calculate the baseline (1993) GWL for each column
df_gwl = pd.read_csv('./data/nonstat/agents_gwl_48.csv', header=None)
df_gwl.columns = ['Year', 'Actual GWL', 'Predicted GWL (M1)', 'Predicted GWL (M2)']

ref_actual_1993 = df_gwl.loc[df_gwl['Year'] == 1993, 'Actual GWL'].iloc[0]
ref_m1_1993 = df_gwl.loc[df_gwl['Year'] == 1993, 'Predicted GWL (M1)'].iloc[0]
ref_m2_1993 = df_gwl.loc[df_gwl['Year'] == 1993, 'Predicted GWL (M2)'].iloc[0]

# 2. Compute the differences relative to the 1993 reference
# convert ft to m
df_gwl['Diff_Actual'] = (df_gwl['Actual GWL'] - ref_actual_1993)*0.3048
df_gwl['Diff_M1'] = (df_gwl['Predicted GWL (M1)'] - ref_m1_1993)*0.3048
df_gwl['Diff_M2'] = (df_gwl['Predicted GWL (M2)'] - ref_m2_1993)*0.3048

# Masks for plotting (2012+ for M1, 2013+ for M2)
mask_m1_gwl = (df_gwl['Year'] >= 2012) & (df_gwl['Year'] <= 2022)
mask_m2_gwl = (df_gwl['Year'] >= 2013) & (df_gwl['Year'] <= 2022)

# --------------------
# 3. COMPUTE R^2 (2014 to 2022) FOR GWU & GWL
# --------------------
# We'll define a mask for data from 2014 onward (through 2022) for both dataframes
mask_2014_2022_gwu = (df_gwu['Year'] >= 2014) & (df_gwu['Year'] <= 2022)
mask_2014_2022_gwl = (df_gwl['Year'] >= 2014) & (df_gwl['Year'] <= 2022)

# a) GWU R^2 for M1 and M2
y_actual_gwu_2014_2022 = df_gwu.loc[mask_2014_2022_gwu, 'Actual GWU']
y_pred_m1_gwu_2014_2022 = df_gwu.loc[mask_2014_2022_gwu, 'Predicted GWU (M1)']
y_pred_m2_gwu_2014_2022 = df_gwu.loc[mask_2014_2022_gwu, 'Predicted GWU (M2)']

r2_gwu_m1 = r2_score(y_actual_gwu_2014_2022, y_pred_m1_gwu_2014_2022)
r2_gwu_m2 = r2_score(y_actual_gwu_2014_2022, y_pred_m2_gwu_2014_2022)

mse_gwu_m1 = mean_squared_error(y_actual_gwu_2014_2022, y_pred_m1_gwu_2014_2022)
mse_gwu_m2 = mean_squared_error(y_actual_gwu_2014_2022, y_pred_m2_gwu_2014_2022)

# b) GWL R^2 for M1 and M2 (using the diff columns)
y_actual_gwl_2014_2022 = df_gwl.loc[mask_2014_2022_gwl, 'Diff_Actual']
y_pred_m1_gwl_2014_2022 = df_gwl.loc[mask_2014_2022_gwl, 'Diff_M1']
y_pred_m2_gwl_2014_2022 = df_gwl.loc[mask_2014_2022_gwl, 'Diff_M2']

r2_gwl_m1 = r2_score(y_actual_gwl_2014_2022, y_pred_m1_gwl_2014_2022)
r2_gwl_m2 = r2_score(y_actual_gwl_2014_2022, y_pred_m2_gwl_2014_2022)

mse_gwl_m1 = mean_squared_error(y_actual_gwl_2014_2022, y_pred_m1_gwl_2014_2022)
mse_gwl_m2 = mean_squared_error(y_actual_gwl_2014_2022, y_pred_m2_gwl_2014_2022)

print("R^2 for GWU (2014-2022):")
print("  M1:", r2_gwu_m1)
print("  M2:", r2_gwu_m2, "\n")

print("MSE for GWU (2014-2022):")
print("  M1:", mse_gwu_m1)
print("  M2:", mse_gwu_m2, "\n")

print("R^2 for GWL (2014-2022):")
print("  M1:", r2_gwl_m1)
print("  M2:", r2_gwl_m2, "\n")

print("MSE for GWL (2014-2022):")
print("  M1:", mse_gwl_m1)
print("  M2:", mse_gwl_m2, "\n")

# --------------------
# 4. PLOTTING
# --------------------
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Arial'

# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

# -- Subplot 1: GWU Lines --
ax1.plot(df_gwu['Year'], df_gwu['Actual GWU']*0.001, color='black', label='Actual GWU')
ax1.plot(df_gwu.loc[mask_m1_gwu, 'Year'], df_gwu.loc[mask_m1_gwu, 'Predicted GWU (M1)']*0.001,
         color='red', label='Predicted GWU (M1)', linestyle='dashed')
ax1.plot(df_gwu.loc[mask_m2_gwu, 'Year'], df_gwu.loc[mask_m2_gwu, 'Predicted GWU (M2)']*0.001,
         color='blue', label='Predicted GWU (M2)', linestyle='dashed')

ax1.set_ylabel('Groundwater Use (m)')
ax1.legend(frameon=False)

# -- Subplot 2: GWL Difference Lines --
ax2.plot(df_gwl['Year'], df_gwl['Diff_Actual'], color='black', label='Actual GWL')
ax2.plot(df_gwl.loc[mask_m1_gwl, 'Year'], df_gwl.loc[mask_m1_gwl, 'Diff_M1'],
         color='red', label='Predicted GWL (M1+RRCA)', linestyle='dashed')
ax2.plot(df_gwl.loc[mask_m2_gwl, 'Year'], df_gwl.loc[mask_m2_gwl, 'Diff_M2'],
         color='blue', label='Predicted GWL (M2+RRCA)', linestyle='dashed')

# 4. Add labels, legend, and show plot
ax2.set_xlabel('Year')
ax2.set_ylabel('Relative Groundwater Level (m)')
# ax2.set_title('Groundwater Level Differences (from 1993)')
ax2.legend(frameon=False)

plt.tight_layout()

# Save the figure as a PDF before showing it
plt.savefig('./data/nonstat/gwu_gwl_comparison_48.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()




