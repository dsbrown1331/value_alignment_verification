import experiment_utils as eutils
import utils
import mdp
import machine_teaching
import copy
import numpy as np
import value_alignment_verification as vav
import random
import sys

def random_weights(num_features):
    return 1.0 - 2.0 * np.random.rand(num_features)

init_seed = 87
num_true_rewards = 100
debug = True

for r_iter in range(num_true_rewards):
    print("="*10, r_iter, "="*10)
    
    ##For this test I want to verify that the ranking-based machine teaching is able to correctly verify whether an agent is value aligned or not.
    #MDP is deterministic with fixed number or rows, cols, and features
    num_rows = 4
    num_cols = 4
    num_features = 4
    num_eval_policies_tries = 100  #Note this isn't how many we'll actually end up with since we reject if same as optimal policy
    initials = [(num_rows // 2, num_cols // 2)]
    terminals = []#[(num_rows-1,num_cols-1)]
    gamma = 0.9
    seed = init_seed + r_iter 
    print("seed", seed)
    np.random.seed(seed)
    random.seed(seed)

    #First let's generate a random MDP
    state_features = eutils.create_random_features_row_col_m(num_rows, num_cols, num_features)
    #print("state features\n",state_features)
    true_weights = random_weights(num_features)
    print("true weights: ", true_weights)  
    true_world = mdp.LinearFeatureGridWorld(state_features, true_weights, initials, terminals, gamma)
    print("rewards")
    true_world.print_rewards()
    print("value function")
    V = mdp.value_iteration(true_world)
    true_world.print_map(V)
    print("mdp features")
    utils.display_onehot_state_features(true_world)
    #find the optimal policy under this MDP
    Qopt = mdp.compute_q_values(true_world, V=V)
    opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt)
    print("optimal policy")
    true_world.print_map(true_world.to_arrows(opt_policy))
    #input()
    #now find a bunch of other optimal policies for the same MDP but with different weight vectors.
    #TODO: I wonder if there is a better way to create these eval policies? 
    # Can we efficiently solve for all of them or should they all be close? (e.g. rewards sampled from gaussian centerd on true reward?)
    world = copy.deepcopy(true_world)
    eval_policies = []
    eval_Qvalues = []
    eval_weights = []
    num_eval_policies = 0
    for i in range(num_eval_policies_tries):
        #print("trying", i)
        #change the reward weights
        eval_weight_vector = random_weights(num_features)
        world.weights = eval_weight_vector
        #find the optimal policy under this MDP
        Qval = mdp.compute_q_values(world)
        eval_policy = mdp.find_optimal_policy(world, Q=Qval)
        #only save if not equal to optimal policy
        if eval_policy != opt_policy and eval_policy not in eval_policies:
            if debug:
                print("found distinct eval policy")
                world.print_map(world.to_arrows(eval_policy))
        
            eval_policies.append(eval_policy)
            eval_Qvalues.append(Qval)
            eval_weights.append(eval_weight_vector)
            num_eval_policies += 1

    print("Running verification tests for {} distinct optimal policies for different reward weights.".format(len(eval_policies)))
    if len(eval_policies) == 0:
        print("only one possible policy. There must be a problem with the features. Can't do verification if only on policy possible!")
        sys.exit()
        


    print("\nGenerating machine teaching test")
    tester = vav.OptimalRankingBasedTester(true_world, debug = debug)
    print("testing true policy")
    verified = tester.is_agent_value_aligned(Qopt)
    #print(verified)
    if not verified:
        print("supposed to verify the optimal policy. This is not right!")
        input()

    correct = 0.0
    for i, Qeval in enumerate(eval_Qvalues):
        if debug:
            print("\ntesting agent", i)
            print("with reward weights:", eval_weights[i])
            print("agent policy")
            world.print_map(world.to_arrows(eval_policies[i]))
            print("compared to ")
            print("optimal policy")
            true_world.print_map(true_world.to_arrows(opt_policy))
            print("true reward weights:", true_weights)
            print("mdp features")
            utils.display_onehot_state_features(true_world)
        verified = tester.is_agent_value_aligned(Qeval)
        #print(verified)
        if verified:
            print("not supposed to be true...")
            input()
        if not verified:
            correct += 1
    print("Accuracy = ", 100.0*correct / num_eval_policies)
    #input()


#teacher = machine_teaching.RankingTeacher(world, debug=False)
#teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)