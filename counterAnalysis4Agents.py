# Conduct counterfactural analysis of the groundwater behavior
# in terms of groundwater irrigation depth (mm) and relative groundwater level (mm).
# Author: Yao Hu
# Date: 12/16/2024


import pandas as pd
import numpy as np
from matplotlib import style, pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore")

agent_gwl_48 = pd.read_csv('./agentdata/nonstat/agents_gwl_48.csv', header=None)
agent_gwl_48.columns = ['Year', 'Actual GWL', 'Predicted GWL']

# agent_gwl_48 = pd.read_csv('./agentdata/nonstat/agents_wu_48.csv', header=None)
# agent_gwl_48.columns = ['Year', 'Actual GWU', 'Predicted GWU']

# find the year with minimum actual gwu
min_year = agent_gwl_48.loc[agent_gwl_48['Actual GWL'].idxmin(), 'Year']
# find the year with maximum actual gwu
max_year = agent_gwl_48.loc[agent_gwl_48['Actual GWL'].idxmax(), 'Year']
# select the agent_gwl_48['Actual GWU'] where the year is equal to 2012
actual_gwu_2012 = agent_gwl_48.loc[agent_gwl_48['Year'] == 2012, 'Actual GWL']

# change the data type of Year to int
agent_gwl_48['Year'] = agent_gwl_48['Year'].astype(int)

# plot the actual and predicted groundwater level for agent 48. The groundwater level is in meter.
# For predicted groundwater level, use the color, red but transparency, alpha=0.5 before 2013
# For predicted groundwater level, use the color, red but dashed line after 2013
# For actual groundwater level, use the color, blue
# Add the title, 'Groundwater Level for Agent 48' and x-axis label, 'Year' and y-axis label, 'Groundwater Level (m)'
# Add the legend, 'Actual GWL' and 'Predicted GWL'
# Save the figure to 'gwl_48.pdf'


mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Arial'

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(agent_gwl_48['Year'], agent_gwl_48['Actual GWL']*1000, color='blue', label='Actual GWL')

# Plot predicted gwl from 2013 onwards using red dashed line
mask2 = agent_gwl_48['Year'] >= 2012
ax.plot(agent_gwl_48.loc[mask2, 'Year'], agent_gwl_48.loc[mask2, 'Predicted GWL']*1000, color='red', linestyle='dashed', label='What-if GWL')

# Add a legend
ax.legend()

# remove the bounding box of the legend
ax.legend(frameon=False)

# Add axis labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Relative Groundwater Level (mm)')

# Show the plot
plt.show()

# Save the figure to 'gwl_48.pdf', which can be edited in Adobe Illustrator
# fig.savefig('./agentdata/gwu_48.pdf', bbox_inches='tight')

# load the data from the csv files, agent_wu.csv without header
# agent_wu.csv has three columns: 'Year', 'Actual GWU' and 'Predicted GWU'





