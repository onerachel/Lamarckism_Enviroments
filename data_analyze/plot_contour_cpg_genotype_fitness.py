#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 10:47:58 2023

@author: LJ
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

path = "/Users/lj/revolve2/Nature_data/"
df = pd.read_csv(path + "summary_all2.csv")
df = df[df['generation'] != 0]

frequency = "5"  # "0", "2", "5"
terrain = "Flat"  # "Flat", "Rugged"

df['generation'] = df['generation'] + 1

# Filter values in a column
Lamarckism = df[df['experiment'] == f'Lamarckism_{terrain}_{frequency}']
Darwinism = df[df['experiment'] == f'Darwinism_{terrain}_{frequency}']

# Set parameters
data = [Darwinism, Lamarckism]

color = ['deepskyblue', 'mediumpurple']  # ,'darkolivegreen', 'limegreen'

# Create separate density plots for Lamarckism and Darwinism
plt.figure(figsize=(12, 6))

# Lamarckism plot
plt.subplot(1, 2, 1)
sns.kdeplot(x=Lamarckism['generation'], y=Lamarckism['after'], fill=True, cmap='Blues', label='Lamarckism', cbar=True)
plt.xlabel('generation')
plt.ylabel('after fitness')
plt.title(f'Lamarckism - {terrain}_{frequency}')
plt.legend()
plt.grid(True, alpha=0.3)

# Darwinism plot
plt.subplot(1, 2, 2)
sns.kdeplot(x=Darwinism['generation'], y=Darwinism['after'], fill=True, cmap='Reds', label='Darwinism', cbar=True)
plt.xlabel('generation')
plt.ylabel('after fitness')
plt.title(f'Darwinism - {terrain}_{frequency}')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


#------------------ 3d plot ------------------#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Dec 23 15:40:58 2023
#
# @author: LJ
# """
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd
# import numpy as np
# import scipy.stats as st
#
# path = "/Users/lj/revolve2/Nature_data/"
# df = pd.read_csv(path + "summary_all.csv")
# df = df[df['generation'] != 0]
#
# frequency = "5"  # "0", "2", "5"
# terrain = "Rugged"  # "Flat", "Rugged"
#
# df['generation'] = df['generation'] + 1
#
# # Filter values in a column
# Lamarckism = df[df['experiment'] == f'Lamarckism_{terrain}_{frequency}']
# Darwinism = df[df['experiment'] == f'Darwinism_{terrain}_{frequency}']
#
# # Calculate the p-value using Welch's t-test
# # _, p_value = ttest_ind(Lamarckism['fitness'], Darwinism['fitness'], equal_var=False)
# # print("p-value: ", p_value)
#
# # Set parameters
# data = [Darwinism, Lamarckism]
#
# color = ['deepskyblue', 'mediumpurple']  # ,'darkolivegreen', 'limegreen'
# std_color = ['lightskyblue', 'mediumpurple']  # ,'aquamarine', 'greenyellow'
#
# # Plot and save
# fig = plt.figure(figsize=(12, 6))
#
# for i, file in enumerate(data):
#     ax = fig.add_subplot(1, 2, i + 1, projection='3d')
#
#     describe = data[i].groupby(['generation', 'after']).describe()['similarity_best']
#     mean = describe['mean'].unstack()
#     std = describe['std'].unstack()
#     max = describe['max'].unstack()
#
#     generations = data[i]['generation'].unique()
#     afters = data[i]['after'].unique()
#
#     # Swap y and z axes by exchanging X and Y
#     Y, X = np.meshgrid(afters, generations)
#
#     standard_error = std / np.sqrt(np.size(describe))
#     confidence_interval = st.t.interval(confidence=0.95, df=len(describe) - 1, loc=mean, scale=standard_error)
#
#     # Swap y and z axes in the plot_surface call
#     # surf = ax.plot_surface(X, Y, mean.T, cmap='viridis', alpha=0.7)
#     surf = ax.plot_surface(X, Y, mean.T, cmap='viridis', facecolors=ax.cmap(ax.norm(mean.T)), alpha=0.7, stride=2)
#
#     ax.set_xlabel('generation')
#     ax.set_ylabel('controller similarity')
#     ax.set_zlabel('fitness')
#
#
#     ax.set_title(data[i]['experiment'].unique()[0])
#
#     # Add color bar
#     fig.colorbar(surf, ax=ax, pad=0.1)
#
# plt.show()
