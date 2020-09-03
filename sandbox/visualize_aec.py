#use the machine teaching example and try to visualize the AEC.

#first sample random reward functions and print out the unique policies:

import src.grid_worlds as grid_worlds
import src.mdp as mdp
import copy
import numpy as np
import random
import src.machine_teaching as machine_teaching
import data_analysis.mdp_family_teaching.twoFeatureFeasibleRegionPlot as plot_aec
import data_analysis.plot_grid as mdp_plot


world = grid_worlds.create_aaai19_toy_world()
state_features = world.features
print(state_features)


def random_weights(num_features):
    rand_n = np.random.randn(num_features)
    l2_ball_weights = rand_n / np.linalg.norm(rand_n)
    return l2_ball_weights
    #return 1.0 - 2.0 * np.random.rand(num_features)

def get_perpendiculars(normal_vec):
    #only works for 2-d return both directions since there are two possibile directions to be perpendicular
    if normal_vec[0] == 0 and normal_vec[1] == 0:
        return [np.array([0,0])]
    elif normal_vec[0] == 0:
        return [np.array([1,0]), np.array([-1,0])]
    elif normal_vec[1] == 0:
        return [np.array([0,1]), np.array([0,-1])]
    else:
        return [np.array([1, -normal_vec[0]/ normal_vec[1]]),np.array([-1,normal_vec[0]/ normal_vec[1]])] 
    
    

num_features = world.get_num_features()
init_seed = 1234
np.random.seed(init_seed)
random.seed(init_seed)

num_eval_policies_tries = 200
debug = True
precision = 0.00001

eval_policies = []
eval_Qvalues = []
eval_weights = []
eval_halfspaces = []
all_halfspaces = []
num_eval_policies = 0
for i in range(num_eval_policies_tries):
    rand_world = copy.deepcopy(world)
    #print("trying", i)
    #change the reward weights
    eval_weight_vector = random_weights(num_features)
    rand_world.weights = eval_weight_vector
    #find the optimal policy under this MDP
    Qval = mdp.compute_q_values(rand_world, eps=precision)
    eval_policy = mdp.find_optimal_policy(rand_world, Q=Qval, epsilon=precision)
    #only save if not equal to optimal policy
    if eval_policy  not in eval_policies:
        if debug:
            print("found distinct eval policy")
            print("weights", eval_weight_vector)

            rand_world.print_map(rand_world.to_arrows(eval_policy))
    
        eval_policies.append(eval_policy)
        eval_Qvalues.append(Qval)
        eval_weights.append(eval_weight_vector)
        teacher = machine_teaching.StateActionRankingTeacher(rand_world, debug=False, epsilon=precision)
        
        
        tests, halfspaces = teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)
        eval_halfspaces.append(halfspaces)
        print(halfspaces)

        #add all the normal vectors to a big list for getting edges later
        for h in halfspaces:
            all_halfspaces.append(h)

        num_eval_policies += 1

print("There are {} distinct optimal policies when sampling randomly".format(len(eval_policies)))

#go through all the halfspaces and do this again:
#now we compute the BEC for each of these. This should give us the boundaries
#we just need to sample from the boundaries to get the rest of the policies.
for h in all_halfspaces:
    for eval_weight_vector in get_perpendiculars(h):
        rand_world = copy.deepcopy(world)
        #print("trying", i)
        #change the reward weights
        rand_world.weights = eval_weight_vector
        #find the optimal policy under this MDP
        Qval = mdp.compute_q_values(rand_world, eps=precision)
        eval_policy = mdp.find_optimal_policy(rand_world, Q=Qval, epsilon=precision)
        #only save if not equal to optimal policy
        if eval_policy  not in eval_policies:
            if debug:
                print("found distinct eval policy")
                print("weights", eval_weight_vector)

                rand_world.print_map(rand_world.to_arrows(eval_policy))
        
            eval_policies.append(eval_policy)
            eval_Qvalues.append(Qval)
            eval_weights.append(eval_weight_vector)
            teacher = machine_teaching.StateActionRankingTeacher(rand_world, debug=False, epsilon=precision)
            
           
            tests, halfspaces = teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)
            eval_halfspaces.append(halfspaces)
            print(halfspaces)

    
            num_eval_policies += 1

#add the zero reward #still not sure if this should combine with the all 1's vector or not....
eval_weight_vector = np.zeros(2)
rand_world = copy.deepcopy(world)
#print("trying", i)
#change the reward weights
rand_world.weights = eval_weight_vector
#find the optimal policy under this MDP
Qval = mdp.compute_q_values(rand_world, eps=precision)
eval_policy = mdp.find_optimal_policy(rand_world, Q=Qval, epsilon=precision)
#only save if not equal to optimal policy
if eval_policy  not in eval_policies:
    if debug:
        print("found distinct eval policy")
        print("weights", eval_weight_vector)

        rand_world.print_map(rand_world.to_arrows(eval_policy))

    eval_policies.append(eval_policy)
    eval_Qvalues.append(Qval)
    eval_weights.append(eval_weight_vector)
    teacher = machine_teaching.StateActionRankingTeacher(rand_world, debug=False, epsilon=precision)
    
    
    tests, halfspaces = teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)
    eval_halfspaces.append(halfspaces)
    print(halfspaces)


print("There are {} distinct optimal policies".format(len(eval_policies)))

from matplotlib.pyplot import cm

# color=cm.get_cmap("Dark2")(np.linspace(0,1,len(eval_policies)))
color=cm.get_cmap("gist_rainbow")(np.linspace(0,1,len(eval_policies)))


#print the AEC's
import matplotlib.pyplot as plt
for i,p in enumerate(eval_policies):
    print(i)
    world.print_map(rand_world.to_arrows(p))
    mdp_plot.plot_optimal_policy_vav(p, state_features, arrow_color='k', filename="/home/dsbrown/Code/value_alignment_verification/data_analysis/figs/aec_toy/policy_bw" + str(i) + ".png")
    halfspaces = eval_halfspaces[i]
    print(halfspaces)
    #plot_aec.plot_feasible_region(halfspaces, range(len(halfspaces)),filename="/home/dsbrown/Code/value_alignment_verification/data_analysis/figs/aec_toy/aec" + str(i) + ".png")
    #plt.show()
    #coloring is weird for hyperplanes as expected.

#Can we print out all the halfspaces we need maybe spoof it? Change policy colors?


###printout blank mdp
# policy = {}
# for r in range(world.rows):
#     for c in range(world.cols):
#         policy[(r,c)] = ""
# mdp_plot.plot_optimal_policy_vav(policy, state_features, filename="/home/dsbrown/Code/value_alignment_verification/data_analysis/figs/aec_toy/blank_mdp.png")
# plt.show()