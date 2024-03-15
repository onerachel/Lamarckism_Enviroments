#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 10:47:58 2023

@author: LJ
"""
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import ttest_ind

# Define the path
path = "/Users/lj/revolve2"

# Parameters
frequency = "5"  # Options: "0", "2", "5"
terrain = "Rugged"  # Options: "Flat", "Rugged"

# Read the file
df = pd.read_csv(path + "/Nature_data/summary_all.csv")
print(df['experiment'].unique())

# Adjusting generation numbering
df['generation'] = df['generation'] + 1

# Filtering the data for both experiments
Lamarckism = df[df['experiment'] == f'Lamarckism_{terrain}_{frequency}']
Darwinism = df[df['experiment'] == f'Darwinism_{terrain}_{frequency}']

# Function to calculate adaptation ratio
def calculate_adaptation_ratio_with_confidence(df):
    df_grouped = df.groupby('generation')['after'].mean()
    adaptation_ratio = df_grouped.shift(1)/ df_grouped
    std_error = df.groupby('generation')['after'].sem()
    confidence = st.t.interval(0.95, len(df_grouped) - 1, loc=adaptation_ratio, scale=std_error)
    return adaptation_ratio, confidence

# Calculate adaptation ratios and their confidence intervals
adaptation_ratio_darwinism, confidence_darwinism = calculate_adaptation_ratio_with_confidence(Darwinism)
adaptation_ratio_lamarckism, confidence_lamarckism = calculate_adaptation_ratio_with_confidence(Lamarckism)

# Colors
color = ['deepskyblue', 'mediumpurple']
std_color = ['lightskyblue', 'mediumpurple']

# Create subplots
figure, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 10))

# First subplot (a)
ax1.plot(adaptation_ratio_darwinism.index, adaptation_ratio_darwinism, color=color[0], label='Darwinism', linewidth=2.0)
ax1.fill_between(adaptation_ratio_darwinism.index, confidence_darwinism[0], confidence_darwinism[1], color=std_color[0], alpha=0.2)
ax1.plot(adaptation_ratio_lamarckism.index, adaptation_ratio_lamarckism, color=color[1], label='Lamarckism', linewidth=2.0)
ax1.fill_between(adaptation_ratio_lamarckism.index, confidence_lamarckism[0], confidence_lamarckism[1], color=std_color[1], alpha=0.2)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylim(0.4, 1.3)
# ax1.plot(adaptation_ratio_darwinism.index, adaptation_ratio_darwinism, color=color[0], label='Darwinism', linewidth=2.0)
# ax1.plot(adaptation_ratio_lamarckism.index, adaptation_ratio_lamarckism, color=color[1], label='Lamarckism', linewidth=2.0)
ax1.set_xlabel('generation', fontsize=14)
ax1.set_ylabel('adaptation rate',fontsize=14)
# ax1.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, fontsize=10)
# ax1.set_title(f'{terrain}_{frequency} Adaptation Ratio')
ax1.grid(True, alpha=0.3)
# ax1.text(-0.05, 1.05, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Second subplot (b)
for i, experiment in enumerate([Darwinism, Lamarckism]):
    describe = experiment.groupby(['generation']).describe()['learning_delta']
    mean = describe['mean']
    std = describe['std']
    confidence_interval = st.t.interval(confidence=0.95, df=len(describe) - 1, loc=mean, scale=std / np.sqrt(np.size(describe)))
    ax2.plot(experiment['generation'].unique(), mean, color=color[i], label=experiment['experiment'].unique()[0], linewidth=2.0)
    ax2.fill_between(experiment['generation'].unique(), confidence_interval[0], confidence_interval[1], color=std_color[i], alpha=0.2)

ax2.set_ylim(0, 2.2)
ax2.tick_params(axis='both', which='major', labelsize=14)  # Set font size of ticks
# ax2.set_xlabel('generation')
ax2.set_ylabel('learning delta', fontsize=14)
ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, fontsize=12)
# ax2.set_title(f'{terrain}_{frequency} Learning Delta')
ax2.grid(True, alpha=0.3)
# ax2.text(-0.05, 1.05, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Show the plot
plt.tight_layout()
plt.savefig(path + f"/Nature_data/plot_images/adaptation_rate_learning_delta_{terrain}_{frequency}.png", bbox_inches='tight')
plt.show()