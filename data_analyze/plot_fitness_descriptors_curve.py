#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2 15:36:30 2023

@author: LJ
"""
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import ttest_ind

path = "/Users/lj/revolve2"

frequency = "0"  # "0", "2", "5"
terrain = "Rugged" # "Flat", "Rugged"
# Read files
df = pd.read_csv(path + "/Nature_data/summary_all.csv")
print(df['experiment'].unique())

df['generation'] = df['generation'] + 1

# Filter values in a column
Lamarckian_Learning = df[df['experiment'] == f'Lamarckism_{terrain}_{frequency}']
Darwinian_Learning = df[df['experiment'] == f'Darwinism_{terrain}_{frequency}']

# Set parameters
data = [Darwinian_Learning, Lamarckian_Learning]
descriptor = ['absolute_size', 'proportion', 'num_bricks', 'rel_num_limbs', 'symmetry', 'branching', 'coverage',
              'rel_num_bricks', 'rel_num_hinges']
color = ['deepskyblue', 'mediumpurple']
std_color = ['lightskyblue', 'mediumpurple']

for j, desc in enumerate(descriptor):
    # Calculate p-value
    t_stat, p_value = ttest_ind(Darwinian_Learning.groupby('generation')[desc].mean(),
                                Lamarckian_Learning.groupby('generation')[desc].mean())

    # Plot and save
    figure, ax = plt.subplots()
    for i, file in enumerate(data):
        describe = data[i].groupby(['generation']).describe()[desc]
        mean = describe['mean']
        std = describe['std']
        max = describe['max']

        standard_error = std / np.sqrt(np.size(describe))
        confidence_interval = st.t.interval(confidence=0.95, df=len(describe) - 1, loc=mean, scale=standard_error)

        ax.plot(data[i]['generation'].unique(), mean, color=color[i], label=data[i]['experiment'].unique()[0],
                linewidth=2.0)
        # standard deviation
        # plt.fill_between(data[i]['generation'].unique(), mean - std, mean + std, color=std_color[i], alpha=0.2)

        # standard error
        # plt.fill_between(data[i]['generation'].unique(), mean - standard_error, mean + standard_error, color=std_color[i], alpha=0.2)

        # 95% confidence interval
        plt.fill_between(data[i]['generation'].unique(), confidence_interval[0], confidence_interval[1],
                         color=std_color[i], alpha=0.2)

        # max
        # ax.scatter(data[i]['generation'].unique(), max, s=8, color=color[i]) #, label=data[i]['experiment'].unique()[0]+" max"

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.set_xlabel('generation')
    ax.set_ylabel(desc)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=2, fontsize=9)
    # ax.set_title(f'p-value: {p_value:.2f}')
    # ax.set_title(f' {terrain}_{frequency}')
    ax.grid(True, alpha=0.3)
    figure.savefig(path + f'/Nature_data/plot_images/descriptor/{terrain}_{frequency}_{desc}.png', bbox_inches='tight')
    plt.show()
