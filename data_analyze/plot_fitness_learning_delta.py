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

path = "/Users/lj/revolve2"

frequency = "0"  # "0", "2", "5"
terrain = "Rugged" # "Flat", "Rugged"
# Read files
df = pd.read_csv(path + "/Nature_data/summary_all.csv")
print(df['experiment'].unique())

df['generation'] = df['generation'] + 1

# Filter values in a column
Lamarckism = df[df['experiment'] == f'Lamarckian_{terrain}_{frequency}']
Darwinism = df[df['experiment'] == f'Darwinian_{terrain}_{frequency}']

# Calculate the p-value using Welch's t-test
# _, p_value = ttest_ind(Lamarckism['fitness'], Darwinism['fitness'], equal_var=False)
# print("p-value: ", p_value)

# Set parameters
data = [Darwinism, Lamarckism]

color = ['deepskyblue', 'mediumpurple']  # ,'darkolivegreen', 'limegreen'
std_color = ['lightskyblue', 'mediumpurple']  # ,'aquamarine', 'greenyellow'

# Plot and save
figure, ax = plt.subplots()
for i, file in enumerate(data):
    describe = data[i].groupby(['generation']).describe()['learning_delta']
    # print(describe)
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
    plt.fill_between(data[i]['generation'].unique(), confidence_interval[0], confidence_interval[1], color=std_color[i],
                     alpha=0.2)

    # max
    # ax.scatter(data[i]['generation'].unique(), max, s=8,
    #            color=color[i])  # , label=data[i]['experiment'].unique()[0]+" max"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ax.set_ylim(0, 2.1)

ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
ax.set_xlabel('generation')
ax.set_ylabel('learning delta')
ax.legend(loc='upper center',
          ncol=2, bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, fontsize=10)
# ax.legend(loc='lower right',
#           ncol=1, fancybox=True, shadow=True, fontsize=9)
# ax.set_title(f'p-value: {p_value:.1e}')
ax.set_title(f'{terrain}_{frequency}')
ax.grid(True, alpha=0.3)
plt.figure(figsize=(3, 100))

figure.savefig(path+"/Nature_data/plot_images/fitness_learning_delta_"+f'{terrain}_{frequency}.png', bbox_inches='tight')

plt.show()
