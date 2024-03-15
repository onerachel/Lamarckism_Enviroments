#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 6 10:47:58 2023

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
terrain = "Flat"  # "Flat", "Rugged"
# Read files
df = pd.read_csv(path + "/Nature_data/summary_all.csv")
print(df['experiment'].unique())

df['generation'] = df['generation'] + 1

# Filter values in a column
Lamarckism = df[df['experiment'] == f'Lamarckism_{terrain}_{frequency}']
Darwinism = df[df['experiment'] == f'Darwinism_{terrain}_{frequency}']

# Calculate the p-value using Welch's t-test
_, p_value = ttest_ind(Lamarckism['morph_tree_dist'], Darwinism['morph_tree_dist'], equal_var=False)
print("p-value: {:.2f}".format(p_value))
# Set parameters
data = [Darwinism, Lamarckism]

color = ['deepskyblue', 'mediumpurple']  # ,'darkolivegreen', 'limegreen'
std_color = ['lightskyblue', 'mediumpurple']  # ,'aquamarine', 'greenyellow'

# Plot and save
figure, ax = plt.subplots()
for i, file in enumerate(data):
    describe = data[i].groupby(['generation']).describe()['morph_tree_dist']
    # print(describe)
    mean = describe['mean']
    std = describe['std']
    max = describe['max']

    standard_error = std / np.sqrt(np.size(describe))
    confidence_interval = st.t.interval(confidence=0.95, df=len(describe) - 1, loc=mean, scale=standard_error)

    ax.plot(data[i]['generation'].unique(), mean, color=color[i], label=data[i]['experiment'].unique()[0],
            linewidth=2.0)

    # 95% confidence interval
    plt.fill_between(data[i]['generation'].unique(), confidence_interval[0], confidence_interval[1], color=std_color[i],
                     alpha=0.2)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
# Set y-axis limits
plt.ylim(8, 18)
ax.set_xlabel('generation')
ax.set_ylabel('morphological diversity')
ax.legend(loc='upper right',
          ncol=1, fancybox=True, shadow=True, fontsize=9)  # 0.5, 1.1, loc='upper center', bbox_to_anchor=(0.4, 1),
# ax.set_title(f'p-value: {p_value:.1e}')
ax.set_title(f' {terrain}_{frequency}')
ax.grid(True, alpha=0.3)
plt.figure(figsize=(3, 100))
figure.savefig(path + f"/Nature_data/plot_images/morphological_diversity_{terrain}_{frequency}.png",
               bbox_inches='tight')

plt.show()
