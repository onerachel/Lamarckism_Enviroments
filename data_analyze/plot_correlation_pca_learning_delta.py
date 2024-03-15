#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 10:47:58 2023

@author: LJ
"""
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your data frame containing the PCA scores and "learning delta" variable
path = "/Users/lj/revolve2"

frequency = "0"  # "0", "2", "5"
terrain = "Rugged"  # "Flat", "Rugged"
df_all = pd.read_csv(path + "/Nature_data/summary_all.csv")
df_all['generation'] = df_all['generation'] + 1

# Filter values in a column
Lamarckism = df_all[df_all['experiment'] == f'Lamarckism_{terrain}_{frequency}']
Darwinism = df_all[df_all['experiment'] == f'Darwinism_{terrain}_{frequency}']

# Set parameters
data = [Darwinism, Lamarckism]

# Create a PCA plot
df = df_all[df_all['experiment'].isin([f'Lamarckism_{terrain}_{frequency}', f'Darwinism_{terrain}_{frequency}'])]
df_pr = df.copy()
df_pr = df_pr[['morphology_' + str(i) for i in range(8)]]
df_pr = (df_pr - df_pr.mean()) / df_pr.std()  # Standardize the data
df_pr = df_pr.rename(columns={col: f'PC{i+1}' for i, col in enumerate(df_pr.columns)})

# Create a data frame with PCA scores and "learning delta"
correlation_data = pd.DataFrame({'PC1': df_pr['PC1'], 'learning_delta': df['learning_delta']})

# Create a scatter plot with a trendline
correlation_plot = sns.lmplot(x='PC1', y='learning_delta', data=correlation_data, height=6, aspect=1.5, ci=None)
correlation_plot.set_axis_labels("PC1", "Learning Delta")
correlation_plot.fig.suptitle("Correlation Plot between PC1 and Learning Delta", y=1.02)

# Save the plot
correlation_plot.savefig("correlation_plot.png")

# Show the plot
plt.show()
