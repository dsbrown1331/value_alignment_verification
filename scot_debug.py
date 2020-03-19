import mdp
import numpy as np
import machine_teaching
import grid_worlds



#world = grid_worlds.create_aaai19_toy_world()
#world = grid_worlds.create_cakmak_task1()
#world = grid_worlds.create_cakmak_task2()
world = grid_worlds.create_cakmak_task4()
num_rollouts = 20
horizon = 100
precision = 0.0001
scot = machine_teaching.SCOT(world, precision, num_rollouts, horizon, debug=False)
demos = scot.get_machine_teaching_demos()
for i,d in enumerate(demos):
    print("demo", i, d)

print("teachable with ", len(demos), "demos!")
