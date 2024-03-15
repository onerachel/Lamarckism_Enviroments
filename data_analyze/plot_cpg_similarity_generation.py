#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 23 15:40:58 2023

@author: LJ
"""
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
from scipy.stats import ttest_ind

path = "/Users/lj/revolve2/Nature_data/"
df = pd.read_csv(path + "summary_all.csv")  #rugged_0 &flat_0 --summary_all
df = df[df['generation'] != 0]

frequency = "5"  # "0", "2", "5"
terrain = "Rugged"  # "Flat", "Rugged"

df['generation'] = df['generation'] + 1

# Filter values in a column
Lamarckian = df[df['experiment'] == f'Lamarckian_{terrain}_{frequency}']
Darwinian = df[df['experiment'] == f'Darwinian_{terrain}_{frequency}']

# Calculate the p-value using Welch's t-test
# _, p_value = ttest_ind(Lamarckian['fitness'], Darwinian['fitness'], equal_var=False)
# print("p-value: ", p_value)

# Set parameters
data = [Darwinian, Lamarckian]

color = ['deepskyblue', 'mediumpurple']  # ,'darkolivegreen', 'limegreen'
std_color = ['lightskyblue', 'mediumpurple']  # ,'aquamarine', 'greenyellow'

# Plot and save
figure, ax = plt.subplots()
for i, file in enumerate(data):
    describe = data[i].groupby(['generation']).describe()['similarity_best']
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
# ax.set_xlim(2, 31)
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
ax.set_xlabel('generation')
ax.set_ylabel('controller similarity')
ax.set_ylim(0.1, 0.95)
ax.legend(loc='upper center',
          ncol=2, bbox_to_anchor=(0.5, 1.1),fancybox=True, shadow=True, fontsize=10)  # 0.5, 1.1, loc='upper center', bbox_to_anchor=(0.4, 1),
# ax.set_title(f'p-value: {p_value:.1e}')
ax.set_title(f'{terrain}_{frequency}')
ax.grid(True, alpha=0.3)
# plt.figure(figsize=(3, 100))

plt.savefig(path+f"plot_images/Controller_similarity_gen_{terrain}_{frequency}.png", bbox_inches='tight')
plt.show()

# # # #--------------------------------------------plot generation & distance (all L AND D merged)--------------------------------------------
# # Filter the DataFrame for Lamarckian and Darwinian separately
# lamarckism_df = df[df['experiment'].str.contains('Lamarckian')]
# darwinism_df = df[df['experiment'].str.contains('Darwinian')]
#
# # Set a smaller figure size
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Create separate lines for "Lamarckian" and "Darwinian"
# sns.lineplot(data=lamarckism_df, x='generation', y='similarity_best', label='Lamarckian', color='mediumpurple')
# sns.lineplot(data=darwinism_df, x='generation', y='similarity_best', label='Darwinian', color='deepskyblue')
#
# # Set the title and axis labels
# # ax.set_title(f'{terrain}_{frequency}', fontsize=11)
# ax.set_xlabel('generation')
# ax.set_ylabel('controller dissimilarity')
# ax.set_title('Controller dissimilarity between parent and child', fontsize=12)
#
# # Rename the legend labels
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles, ['Darwinian+Learning','Lamarckian+Learning'])
#
# plt.savefig(path+f"plot_images/Controller_dissimilarity_gen_all.png", bbox_inches='tight')
# plt.show()

