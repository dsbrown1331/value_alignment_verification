import experiment_utils as eutils
import mdp
import machine_teaching
import copy
import numpy as np
import value_alignment_verification as vav
import random

def random_weights(num_features):
    return 1.0 - 2.0 * np.random.rand(num_features)

##For this test I want to verify that the ranking-based machine teaching is able to correctly verify whether an agent is value aligned or not.
#MDP is deterministic with fixed number or rows, cols, and features
num_rows = 5
num_cols = 5
num_features = 3
num_eval_policies_tries = 10  #Note this isn't how many we'll actually end up with since we reject if same as optimal policy
initials = [(num_rows // 2, num_cols // 2)]
terminals = []#[(num_rows-1,num_cols-1)]
gamma = 0.95
seed = 12
np.random.seed(seed)
random.seed(seed)

#First let's generate a random MDP
state_features = eutils.create_random_features_row_col_m(num_rows, num_cols, num_features)
true_weights = random_weights(num_features)
    
true_world = mdp.LinearFeatureGridWorld(state_features, true_weights, initials, terminals, gamma)

#find the optimal policy under this MDP
Qopt = mdp.compute_q_values(true_world)
opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt)
print("optimal policy")
true_world.print_map(true_world.to_arrows(opt_policy))

#now find a bunch of other optimal policies for the same MDP but with different weight vectors.
#TODO: I wonder if there is a better way to create these eval policies? 
# Can we efficiently solve for all of them or should they all be close? (e.g. rewards sampled from gaussian centerd on true reward?)
world = copy.deepcopy(true_world)
eval_policies = []
eval_Qvalues = []
num_eval_policies = 0
for i in range(num_eval_policies_tries):
    print("eval", i)
    #change the reward weights
    world.weights = random_weights(num_features)
    #find the optimal policy under this MDP
    Qval = mdp.compute_q_values(world)
    eval_policy = mdp.find_optimal_policy(world, Q=Qval)
    print("eval policy")
    world.print_map(world.to_arrows(eval_policy))
    #only save if not equal to optimal policy
    if eval_policy != opt_policy:
        eval_policies.append(eval_policy)
        eval_Qvalues.append(Qval)
        num_eval_policies += 1

print("There are {} distinct optimal policies".format(len(eval_policies)))


tester = vav.RankingBasedTester(true_world)
print("testing true policy")
verified = tester.is_agent_value_aligned(Qopt)
print(verified)

correct = 0.0
for i, Qeval in enumerate(eval_Qvalues):
    print("testing agent", i)
    print("agent policy")
    world.print_map(world.to_arrows(eval_policies[i]))
    print("vs")
    print("optimal policy")
    true_world.print_map(true_world.to_arrows(opt_policy))
    verified = tester.is_agent_value_aligned(Qeval)
    print(verified)
    if verified:
        print("not supposed to be true...")
        input()
    if not verified:
        correct += 1
print("Accuracy = ", 100.0*correct / num_eval_policies)


#teacher = machine_teaching.RankingTeacher(world, debug=False)
#teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)