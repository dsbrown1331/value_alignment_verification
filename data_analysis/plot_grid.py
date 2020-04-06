import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def plot_dashed_arrow(state, width, ax, direction):
    print("plotting dashed arrow", direction)
    h_length = 0.15
    shaft_length = 0.4
    
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -shaft_length + h_length
    elif direction is 'left':
        x_end = -shaft_length + h_length
        y_end = 0
    elif direction is 'right':
        x_end = shaft_length - h_length
        y_end = 0
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=None, head_length=None, fc='k', ec='k',linewidth=4, linestyle=':',fill=False) 
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = h_length
        y_coord += shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -h_length
        y_coord += -shaft_length + h_length
    elif direction is 'left':
        x_end = -h_length
        y_end = 0
        x_coord += -shaft_length + h_length
    elif direction is 'right':
        x_end = h_length
        y_end = 0
        x_coord += shaft_length - h_length
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2, head_length=h_length, fc='k', ec='k',linewidth=4, fill=False,length_includes_head = True) 

def plot_arrow(state, width, ax, direction):
    print("plotting arrow", direction)
    h_length = 0.15
    shaft_length = 0.4
    
    #convert state to coords where (0,0) is top left
    x_coord = state % width
    y_coord = state // width
    print(x_coord, y_coord)
    if direction is 'down':
        x_end = 0
        y_end = shaft_length - h_length
    elif direction is 'up':
        x_end = 0
        y_end = -shaft_length + h_length
    elif direction is 'left':
        x_end = -shaft_length + h_length
        y_end = 0
    elif direction is 'right':
        x_end = shaft_length - h_length
        y_end = 0
    else:
        print("ERROR: ", direction, " is not a valid action")
        return
    print(x_end, y_end)
    ax.arrow(x_coord, y_coord, x_end, y_end, head_width=0.2, head_length=h_length, fc='k', ec='k',linewidth=4) 

def plot_dot(state, width, ax):
    ax.plot(state % width, state // width, 'ko',markersize=10)
    

def plot_optimal_policy(pi, feature_mat):
    plt.figure()

    ax = plt.axes() 
    count = 0
    rows,cols = len(pi), len(pi[0])
    for line in pi:
        for el in line:
            print("optimal action", el)
            # could be a stochastic policy with more than one optimal action
            for char in el:
                print(char)
                if char is "^":
                    plot_arrow(count, cols, ax, "up")
                elif char is "v":
                    plot_arrow(count, cols, ax, "down")
                elif char is ">":
                    plot_arrow(count, cols, ax, "right")
                elif char is "<":
                    plot_arrow(count, cols, ax, "left")
                elif char is ".":
                    plot_dot(count, cols, ax)
                elif el is "w":
                    #wall
                    pass
            count += 1

    
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    cmap = colors.ListedColormap(['black','white','tab:blue','tab:red','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')
    # Add the grid
    ax = plt.gca()
    # Minor ticks
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    #remove ticks
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off') # labels along the bottom edge are off

    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.show()
    
    
def plot_test_query(state, better_action, worse_action, feature_mat, equal_pref = False):

    plt.figure()
    ax = plt.axes() 
    count = 0
    rows,cols = len(feature_mat), len(feature_mat[0])
    if better_action is "^":
        plot_arrow(state, cols, ax, "up")
    elif better_action is "v":
        plot_arrow(state, cols, ax, "down")
    elif better_action is ">":
        plot_arrow(state, cols, ax, "right")
    elif better_action is "<":
        plot_arrow(state, cols, ax, "left")
        
    if equal_pref:
        if worse_action is "^":
            plot_arrow(state, cols, ax, "up")
        elif worse_action is "v":
            plot_arrow(state, cols, ax, "down")
        elif worse_action is ">":
            plot_arrow(state, cols, ax, "right")
        elif worse_action is "<":
            plot_arrow(state, cols, ax, "left")

    
    else:
    
        if worse_action is "^":
            plot_dashed_arrow(state, cols, ax, "up")
        elif worse_action is "v":
            plot_dashed_arrow(state, cols, ax, "down")
        elif worse_action is ">":
            plot_dashed_arrow(state, cols, ax, "right")
        elif worse_action is "<":
            plot_dashed_arrow(state, cols, ax, "left")

    
    mat = [[0 if fvec is None else fvec.index(1)+1 for fvec in row] for row in feature_mat]
    #convert feature_mat into colors
    #heatmap =  plt.imshow(mat, cmap="Reds", interpolation='none', aspect='equal')
    cmap = colors.ListedColormap(['black','white','tab:blue','tab:red','tab:green','tab:purple', 'tab:orange', 'tab:gray', 'tab:cyan'])
    plt.imshow(mat, cmap=cmap, interpolation='none', aspect='equal')
    # Add the grid
    ax = plt.gca()
    # Minor ticks
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True);
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True);
    ax.grid(which='minor', axis='both', linestyle='-', linewidth=5, color='k')
    #remove ticks
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off') # labels along the bottom edge are off

    #cbar = plt.colorbar(heatmap)
    #cbar.ax.tick_params(labelsize=20) 
    plt.show()
    
    
if __name__=="__main__":
    pi = [['v', '^><','.'],['<>v','<','>'],['<>^v','v' ,'^']]
    feature_mat = [[(1,0,0),(0,1,0),(0,0,1)],[(0,0,0,1),(0,0,0,0,1),(0,0,0,0,0,1)],[(0,0,0,0,0,0,1), (0,0,0,0,0,0,0,1),None]  ]      
    plot_optimal_policy(pi, feature_mat)
    
    state = 3  #the integer value of state starting from top left and reading left to right, top to bottom.
    better_action = "v"
    worse_action = "<"
    #plot the optimal test query, where the right answer is bolded  (add equal_pref=True argument if both are equally good)
    plot_test_query(state, better_action, worse_action, feature_mat)
    
    state = 4  #the integer value of state starting from top left and reading left to right, top to bottom.
    better_action = "v"
    worse_action = "<"
    plot_test_query(state, better_action, worse_action, feature_mat, equal_pref = True)

