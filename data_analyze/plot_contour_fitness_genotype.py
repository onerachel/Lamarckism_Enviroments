#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 3 11:47:58 2023

@author: LJ
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
path = "/Users/lj/revolve2/Nature_data/"
df = pd.read_csv(path + "summary_all.csv")
df = df[df['generation'] != 0]

frequency = "0"  # "0", "2", "5"
terrain = "Rugged"  # "Flat", "Rugged"

df['generation'] = df['generation'] + 1

# Filter values in a column
Lamarckian_Learning = df[df['experiment'] == f'Lamarckism_{terrain}_{frequency}']
Darwinian_Learning = df[df['experiment'] == f'Darwinism_{terrain}_{frequency}']

experiments = [('Lamarckism', Lamarckian_Learning), ('Darwinism', Darwinian_Learning)]

# Define function to create contour plot
def create_contour_plot(experiment_name, data):
    # plt.figure(figsize=(3.31, 3))
    plt.figure(figsize=(5, 4.5))

    fitnesses = data['after']
    x = data['similarity_best']
    y = data['morph_tree_dist']

    x_range = np.arange(min(x), max(x), 0.01)
    y_range = np.arange(min(y), max(y), 0.001)
    xx, yy = np.meshgrid(x_range, y_range, sparse=True)
    f = np.zeros((xx.size, yy.size))

    sigma = 0.1
    for ind in range(len(fitnesses)):
        gx = np.exp(-((x_range - x.iloc[ind]) / (max(x) - min(x))) ** 2 / (2 * sigma ** 2))
        gy = np.exp(-((y_range - y.iloc[ind]) / (max(y) - min(y))) ** 2 / (2 * sigma ** 2))
        g = np.outer(gx, gy)
        f += g * (fitnesses.iloc[ind] - np.sum(fitnesses) / len(fitnesses))

    f -= np.min(f)
    f /= np.max(f)
    opt_range = np.max(fitnesses) - np.min(fitnesses)
    f = f * opt_range + np.min(fitnesses)

    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    # Option 1: without the same color scale
    # contour = plt.contourf(x_range, y_range, f.T)

    # Option2: set same color scale to be between -8 and 64
    levels = np.linspace(-0.5, 2.6) #(-0.8, 2.8) for rotation
    contour = plt.contourf(x_range, y_range, f.T, levels=levels)
    ticks = np.arange(-0.5, 2.6, 0.5)  # (-0.8, 2.8, 0.4) Set colorbar ticks from -8 to 64 without decimals, in steps of 8
    plt.colorbar(contour, ticks=ticks)
    plt.plot(x, y, 'ko', ms=0.8)

    plt.xlabel('controller similarity', fontsize=10)
    plt.ylabel('morphological dissimilarity (td)', fontsize=10)
    plt.title(f'{experiment_name}_{terrain}_{frequency}', fontsize=10)
    plt.grid(True)

    plt.tight_layout()

    #Save plot as PDF file
    plt.savefig(path+f"plot_images/{experiment_name}_{terrain}_{frequency}_contour_fitness_morph_cpg3.png", bbox_inches='tight')


# Create contour plots for each experiment
for experiment_name, data in experiments:
    create_contour_plot(experiment_name, data)

plt.show()