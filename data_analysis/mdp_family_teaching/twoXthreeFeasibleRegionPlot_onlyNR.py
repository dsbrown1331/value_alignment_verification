# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:46:49 2018

@author: daniel
"""


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from matplotlib import collections  as mc

##data for full policy
halfspaces = [              [ 0.9304349,  -0.36645723],
              [-1.,          0.        ]]
non_redundants = [0,1]
##data for demo
#halfspaces = [[ -1., 0.],
#              [ 1., -1.]]
#non_redundants = [0,1]

plt.figure(1)
colors = ['b','r','g','c','m']
one_color= 'b'
cnt = 0
fillamount = 0.2
for h in halfspaces:
    #plot line with correct normal vector
    normal_dir = np.sign(np.dot(h, [0,1]))
    if h[1] != 0:
        plt.plot([-1, 1], [h[0]/h[1], -h[0]/h[1]],color=one_color, linestyle= '--')
        plt.fill_between([-1, 1], [h[0]/h[1], -h[0]/h[1]], 
                         [h[0]/h[1] + normal_dir*100, -h[0]/h[1] + normal_dir *100], 
                          facecolor=one_color, interpolate=True, alpha=fillamount)
    else:
        plt.plot([0, 0],[-1, 1],color=one_color, linestyle='--')
        if h[0] < 0:
            plt.fill_between([-1, 0], [-1, -1], [1, 1], facecolor=one_color, interpolate=True, alpha=fillamount)
        else:
            plt.fill_between([0, 1], [-1, -1], [1, 1], facecolor=one_color, interpolate=True, alpha=fillamount)
    #plot point in feasible region of same color
    #plt.plot(h[0],h[1],'o',color=colors[cnt])
    
    cnt += 1
#plot non_redundants in solid lines
  
for i in non_redundants:
    h = halfspaces[i]
    normal_dir = np.sign(np.dot(h, [0,1]))
    if h[1] != 0:
        plt.plot([-1, 1], [h[0]/h[1], -h[0]/h[1]],color=one_color, linestyle='-', linewidth = 2)
    else:
        plt.plot([0, 0],[-1, 1],color=one_color, linestyle='-', linewidth = 2)
    
#plt.fill_between([-1, 0],[-1, 0],[-1, -1], facecolor='none', interpolate=True, hatch='+')
##uncomment for full policy feasible
h = halfspaces[0]
plt.fill_between([h[1]/h[0], 0],[-1, 0],[-1, -1], facecolor='none', interpolate=True, hatch='+')
plt.fill_between([h[1]/h[0], 0],[-1, 0],[-1, -1], facecolor=one_color, interpolate=True, alpha=fillamount)

plt.axis([-1,1,-1,1])
plt.xlabel("$w_0$", fontsize=50)
plt.ylabel("$w_1$", fontsize=50)
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30) 
plt.tight_layout()
#for i in range(1000):
#    rand_sample = [2*np.random.rand() - 1, 2*np.random.rand() - 1]
#    #check if in intersection
#    in_intersection = True
#    for h in halfspaces:
#        if np.dot(h,rand_sample) < 0:
#            in_intersection = False
#    if in_intersection:
#        plt.plot(rand_sample[0],rand_sample[1],'.')
        
plt.show()



#x w0 + y w1 >= 0
#y = -w0/w1 x plug in x = -1 to get one endpoint and x = 1 to get the other
#
#lines = []
#for line in non_dup_constraints:
#    line_seg = []
#    line_seq.append(tuple())
#    lines.append()
#c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
#
#lc = mc.LineCollection(lines, colors=c, linewidths=2)
#fig, ax = pl.subplots()
#ax.add_collection(lc)
#ax.autoscale()
#ax.margins(0.1)
