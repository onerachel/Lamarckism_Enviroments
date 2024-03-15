#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 10:47:58 2023

@author: LJ
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

path = "/Users/lj/revolve2"

# Read files
df = pd.read_csv(f"{path}/Nature_data/survivors.csv")
df['generation'] = df['generation'] -1

experiments = ['Flat_2', 'Rugged_2', 'Flat_5', 'Rugged_5']

# Initialize a dictionary to store the mean of ratios for each experiment
data_dict = {'Experiment': [], 'Lamarckism_Mean_Ratio': [], 'Darwinism_Mean_Ratio': [], 'P-value': []}

# Calculate ratios for each experiment and store in the dictionary
for experiment_type in experiments:
    lamarckism_name = f"Lamarckism_{experiment_type}"
    darwinism_name = f"Darwinism_{experiment_type}"

    lamarckism_df = df[df['experiment'] == lamarckism_name]
    darwinism_df = df[df['experiment'] == darwinism_name]

    # Define generation pairs to use based on experiment type
    if experiment_type == 'Flat_2':
        generation_pairs = [(10, 11)]
    elif experiment_type == 'Flat_5':
        generation_pairs = [(5, 6), (15, 16),(25, 26)]
    elif experiment_type == 'Rugged_2':
        generation_pairs = [(20, 21)]
    elif experiment_type == 'Rugged_5':
        generation_pairs = [(10, 11),(20, 21)]

    lamarckism_run_ratios = []
    darwinism_run_ratios = []

    # Iterate over the specified generation pairs and calculate fitness ratios for each run
    for earlier_gen, later_gen in generation_pairs:
        run_ids = set(lamarckism_df['run'].unique()).union(set(darwinism_df['run'].unique()))

        for run_id in run_ids:
            # Lamarckism
            lamarckism_earlier = lamarckism_df[
                (lamarckism_df['generation'] == earlier_gen) & (lamarckism_df['run'] == run_id)]
            lamarckism_later = lamarckism_df[
                (lamarckism_df['generation'] == later_gen) & (lamarckism_df['run'] == run_id)]
            if not lamarckism_earlier.empty and not lamarckism_later.empty:
                lamarckism_run_ratio = lamarckism_later['fitness'].mean() / lamarckism_earlier['fitness'].mean()
                lamarckism_run_ratios.append(lamarckism_run_ratio)

            # Darwinism
            darwinism_earlier = darwinism_df[
                (darwinism_df['generation'] == earlier_gen) & (darwinism_df['run'] == run_id)]
            darwinism_later = darwinism_df[(darwinism_df['generation'] == later_gen) & (darwinism_df['run'] == run_id)]
            if not darwinism_earlier.empty and not darwinism_later.empty:
                darwinism_run_ratio = darwinism_later['fitness'].mean() / darwinism_earlier['fitness'].mean()
                darwinism_run_ratios.append(darwinism_run_ratio)

    # Calculate the mean of run ratios for Lamarckism and Darwinism
    mean_lamarckism_ratio = np.mean(lamarckism_run_ratios) if lamarckism_run_ratios else np.nan
    mean_darwinism_ratio = np.mean(darwinism_run_ratios) if darwinism_run_ratios else np.nan

    # Calculate p-value between Lamarckism and Darwinism run ratios
    t_stat, p_value = ttest_ind(lamarckism_run_ratios, darwinism_run_ratios, nan_policy='omit')

    # Store the mean of ratios and p-value in the dictionary
    data_dict['Experiment'].append(experiment_type)
    data_dict['Lamarckism_Mean_Ratio'].append(mean_lamarckism_ratio)
    data_dict['Darwinism_Mean_Ratio'].append(mean_darwinism_ratio)
    data_dict['P-value'].append(p_value)

# Create a DataFrame from the dictionary
ratios_df = pd.DataFrame(data_dict).set_index('Experiment')

# Plot the mean ratios for each experiment as grouped bars
ax = ratios_df[['Lamarckism_Mean_Ratio', 'Darwinism_Mean_Ratio']].plot(kind='bar', color=['mediumpurple', 'lightblue'],
                                                                       figsize=(10, 8))
plt.ylabel('transferability', fontsize=25)
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['Lamarckian', 'Darwinian'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True,
           shadow=True, fontsize=20)
ax.set_xlabel('')
plt.ylim(0, 1.4)  # Adjust the upper limit as needed

# Annotate plot with p-values
for i, (experiment, row) in enumerate(ratios_df.iterrows()):
    # Coordinates for the annotation line
    x1, x2 = i - 0.15, i + 0.15  # adjust these values to change the position of line
    y, h, col = ratios_df[['Lamarckism_Mean_Ratio', 'Darwinism_Mean_Ratio']].values.max() * 1.06, ratios_df[
        ['Lamarckism_Mean_Ratio', 'Darwinism_Mean_Ratio']].values.max() * 0.02, 'k'

    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h+0.01, f"{row['P-value']:.2e}", ha='center', va='bottom', color=col, fontsize=14)

# Adding value labels on top of each bar
for i, (index, row) in enumerate(ratios_df.iterrows()):
    lamarckism_ratio, darwinism_ratio, p_value = row
    plt.text(i - 0.15, lamarckism_ratio + 0.01, f'{lamarckism_ratio:.2f}', ha='center', va='bottom', color='black',
             fontsize=14)
    plt.text(i + 0.15, darwinism_ratio + 0.01, f'{darwinism_ratio:.2f}', ha='center', va='bottom', color='black',
             fontsize=14)

plt.tight_layout()
plt.savefig(path+"/Nature_data/plot_images/survivor_transferability_easy_hard-1.png", bbox_inches='tight')
plt.show()
