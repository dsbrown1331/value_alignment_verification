
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import sys
plt.figure()

mat =[[0,0],[1,2]]
rows, cols = len(mat), len(mat[0])
#convert feature_mat into colors
#heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
im = plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')

ax = plt.gca()

ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
#ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
# Gridlines based on minor ticks
ax.grid(which='minor', color='k', linestyle='-', linewidth=5)
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_locator(plt.NullLocator())

# mat =[[0,0],[1,2]]
# rows, cols = len(mat), len(mat[0])
# #convert feature_mat into colors
# #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
# cmap = colors.ListedColormap(['white','tab:red','tab:blue','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
# plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')
# # Add the grid
# ax = plt.gca()
# # Minor ticks
# ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
# ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
# ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
# plt.axis('off')
# ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
# #remove ticks
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     left='off',
#     right='off',
#     labelbottom='off',
#     labelleft='off') # labels along the bottom edge are off

#cbar = plt.colorbar(heatmap)
#cbar.ax.tick_params(labelsize=20) 
#plt.show()

plt.show()