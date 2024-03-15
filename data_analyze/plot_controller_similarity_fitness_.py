#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 10:47:58 2023

@author: LJ
"""

from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp

# Set Seaborn context for larger fonts
sns.set_context("notebook", font_scale=1.5)

path = "/Users/lj/revolve2/Nature_data/"
df = pd.read_csv(path + "summary_all.csv")  # best_parent_distance.csv

# --------------------------------------------plot fitness & dist correlation - dot plots --------------------------------------------
frequency = "0"  # "0", "2", "5"
terrain = "Rugged"  # "Flat", "Rugged"

# Filter the DataFrame
filtered_df = df[(df['experiment'] == f'Lamarckian_{terrain}_{frequency}') | (df['experiment'] == f'Darwinian_{terrain}_{frequency}')]

# Group the filtered data by "generation", "experiment", and "individual_id"
grouped = filtered_df.groupby(['generation', 'experiment', 'individual_id']).agg({'after': 'mean', 'similarity_best': 'mean'}).reset_index()

# Separate the data for Lamarckian and Darwinian experiments
lamarckian = grouped[grouped['experiment'] == f'Lamarckian_{terrain}_{frequency}']
darwinian = grouped[grouped['experiment'] == f'Darwinian_{terrain}_{frequency}']

# Plotting using plt.scatter for more control over transparency
plt.figure(figsize=(8.8, 8))  # Set the figure size to 8.8 inches width and 8 inches height

# Darwinian scatter plot with transparency
plt.scatter(darwinian['similarity_best'], darwinian['after'], color='blue', label=f'Darwinian_{terrain}_{frequency}')

# Lamarckian scatter plot with transparency
plt.scatter(lamarckian['similarity_best'], lamarckian['after'], color='orange', alpha=0.5, label=f'Lamarckian_{terrain}_{frequency}')

# Add regression lines
sns.regplot(data=lamarckian, x='similarity_best', y='after', scatter=False, color='orange')
sns.regplot(data=darwinian, x='similarity_best', y='after', scatter=False, color='blue')

# Set the title and axis labels with larger font sizes
plt.xlabel('controller similarity', fontsize=20)
plt.ylabel('fitness', fontsize=20)

# Add the legend
plt.legend(
    loc='upper center',
    ncol=2,
    bbox_to_anchor=(0.5, 1.07),
    fancybox=True,
    shadow=True,
    fontsize=15
)

# Remove the top and right spines
plt.gca().spines['top'].set_visible(False)  # Hide the top spine
plt.gca().spines['right'].set_visible(False)  # Hide the right spine

# Saving the plot
plt.savefig(path + f"plot_images/controller_similarity_fitness_{terrain}_{frequency}.png", bbox_inches='tight', dpi=110)

plt.show()
