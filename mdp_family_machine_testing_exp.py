import experiment_utils as eutils
import utils
import mdp
import machine_teaching
import copy
import numpy as np
import value_alignment_verification as vav
import alignment_heuristics as ah
import random
import sys
import gridNxNexhaustive as mdp_gen
import data_analysis.mdp_family_teaching.plotFeasibleRewards

def random_weights(num_features):
    rand_n = np.random.randn(num_features)
    l2_ball_weights = rand_n / np.linalg.norm(rand_n)
    return l2_ball_weights
    #return 1.0 - 2.0 * np.random.rand(num_features)

init_seed = 1234

num_features = 2
grid_length = 2
num_cols = grid_length
num_rows = grid_length
initials = [(i,j) for i in range(num_rows) for j in range(num_cols)]

gamma = 0.95
use_terminal = True
max_mdps = 5000
debug = False
precision = 0.00001
true_weights = [-0.1,-0.9] #escape but avoid second feature

np.random.seed(init_seed)
random.seed(init_seed)

#find the intersection of the AEC for a family of MDPs

#state_features = eutils.create_random_features_row_col_m(num_rows, num_cols, num_features)
#print("state features\n",state_features)


unique_mdps = mdp_gen.get_all_unique_mdps(num_features, grid_length, use_terminal, max_mdps)
mdp_family = []
#for each grid set up an MDP env
for mdp_grid, term_grid in unique_mdps:
    print("--"*10)
    state_features = mdp_grid
    terminals = mdp_gen.get_terminals_from_grid(term_grid)
    #print("state features\n",state_features)
    state_features = mdp_gen.categorical_to_one_hot_features(state_features, num_features)
    print('one hot features', state_features)

    world = mdp.LinearFeatureGridWorld(state_features, true_weights, initials, terminals, gamma)
    mdp_family.append(world)

family_teacher = machine_teaching.MdpFamilyTeacher(mdp_family, precision, debug)
mdp_set_cover = family_teacher.get_machine_teaching_mdps()

print("SOLUTION:::::")
for true_world in mdp_set_cover:
    print()
    V = mdp.value_iteration(true_world, epsilon=precision)
    Qopt = mdp.compute_q_values(true_world, V=V, eps=precision)
    opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt, epsilon=precision)


    print("true weights: ", true_weights)  

    print("rewards")
    true_world.print_rewards()
    print("value function")

    true_world.print_map(V)
    print("mdp features")
    utils.display_onehot_state_features(true_world)

    print("optimal policy")
    true_world.print_map(true_world.to_arrows(opt_policy))

    # #run value alignment verification test gen to get halfspaces
    # tester = vav.HalfspaceVerificationTester(true_world, debug = False, precision=precision)
    # size_verification_test = tester.get_size_verification_test()
    # print(tester.halfspaces)

# init_seed = 1234
# num_trials = 50  #number of mdps with random rewards to try
# num_eval_policies_tries = 50

# #scot params
# num_rollouts = 20
# rollout_length = 40

# debug = False
# precision = 0.00001
# num_rows_list = [4,8,16]
# num_cols_list = [4,8,16]
# num_features_list = [2,3,4,5,6,7,8]
# #verifier_list = ["scot", "optimal_action","state-value-critical-1.0","state-value-critical-0.5","state-value-critical-0.1", "ranking-halfspace"]
# verifier_list = ["state-value-critical-0.7","state-value-critical-0.2","state-value-critical-0.01"]
# exp_data_dir = "./experiment_data/"


# for num_features in num_features_list:
#     for num_rows in num_rows_list:
#         num_cols = num_rows #keep it square grid for  now

#         result_writers = []
#         for i, verifier_name in enumerate(verifier_list):
#             filename = "{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
#             print("writing to", filename)
#             result_writers.append(open(exp_data_dir + filename,'w'))

#         for r_iter in range(num_trials):
#             print("="*10, r_iter, "="*10)
#             print("features", num_features, "num_rows", num_rows)
#             ##For this test I want to verify that the ranking-based machine teaching is able to correctly verify whether an agent is value aligned or not.
#             #MDP is deterministic with fixed number or rows, cols, and features
#             #try a variable number of eval policies since bigger domains can have more possible policies (this is just a heuristic to make sure we try a lot but not as many for really small mdps)
#             # 2 * num_features * num_rows * num_cols #Note this isn't how many we'll actually end up with since we reject if same as optimal policy
#             initials = [(i,j) for i in range(num_rows) for j in range(num_cols)]
#             terminals = []#[(num_rows-1,num_cols-1)]
#             gamma = 0.9
#             seed = init_seed + r_iter 
#             print("seed", seed)
#             np.random.seed(seed)
#             random.seed(seed)

#             #First let's generate a random MDP
#             state_features = eutils.create_random_features_row_col_m(num_rows, num_cols, num_features)
#             #print("state features\n",state_features)
#             true_weights = random_weights(num_features)
#             true_world = mdp.LinearFeatureGridWorld(state_features, true_weights, initials, terminals, gamma)
#             V = mdp.value_iteration(true_world, epsilon=precision)
#             Qopt = mdp.compute_q_values(true_world, V=V, eps=precision)
#             opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt, epsilon=precision)
            
#             if debug:
#                 print("true weights: ", true_weights)  
            
#                 print("rewards")
#                 true_world.print_rewards()
#                 print("value function")
            
#                 true_world.print_map(V)
#                 print("mdp features")
#                 utils.display_onehot_state_features(true_world)
            
#                 print("optimal policy")
#                 true_world.print_map(true_world.to_arrows(opt_policy))
            
#             #now find a bunch of other optimal policies for the same MDP but with different weight vectors.
#             world = copy.deepcopy(true_world)
#             eval_policies = []
#             eval_Qvalues = []
#             eval_weights = []
#             num_eval_policies = 0
#             for i in range(num_eval_policies_tries):
#                 #print("trying", i)
#                 #change the reward weights
#                 eval_weight_vector = random_weights(num_features)
#                 world.weights = eval_weight_vector
#                 #find the optimal policy under this MDP
#                 Qval = mdp.compute_q_values(world, eps=precision)
#                 eval_policy = mdp.find_optimal_policy(world, Q=Qval, epsilon=precision)
#                 #only save if not equal to optimal policy
#                 if eval_policy != opt_policy and eval_policy not in eval_policies:
#                     if debug:
#                         print("found distinct eval policy")
#                         world.print_map(world.to_arrows(eval_policy))
                
#                     eval_policies.append(eval_policy)
#                     eval_Qvalues.append(Qval)
#                     eval_weights.append(eval_weight_vector)
#                     num_eval_policies += 1

#             print("There are {} distinct optimal policies".format(len(eval_policies)))
#             if len(eval_policies) == 0:
#                 print("The only possible policy is the optimal policy. There must be a problem with the features. Can't do verification if only on policy possible!")
#                 sys.exit()
                

#             print()
#             print("Generating verification tests")

#             #TODO: run through all the verifiers and create tests for current MDP
#             #TODO: develop a common interface that they all implement 

            
#             #TODO: have a list of names
#             for vindx, verifier_name in enumerate(verifier_list):
#                 tester = None
#                 size_verification_test = None

#                 if "state-value-critical-" in verifier_name:
#                     critical_value_thresh = float(verifier_name[len("state-value-critical-"):])
#                     #print("critical value", critical_value_thresh)
#                     tester = ah.CriticalStateActionValueVerifier(true_world, critical_value_thresh, precision=precision)
#                 elif verifier_name == "ranking-halfspace":
#                     tester = vav.HalfspaceVerificationTester(true_world, debug = debug, precision=precision)

#                 elif verifier_name == "state-optimal-action_ranker":
#                     tester = vav.RankingBasedTester(true_world, precision, debug=debug)

#                 elif verifier_name == "optimal_action":
#                     tester = vav.OptimalRankingBasedTester(true_world, precision, debug=debug)

#                 elif verifier_name == "optimal_action_allquestions":
#                     tester = vav.OptimalRankingBasedTesterAll(true_world, precision, debug=debug)

#                 elif verifier_name == "scot":
#                     tester = vav.SCOTVerificationTester(true_world, precision, num_rollouts, rollout_length, debug=debug)
                
#                 else:
#                     print("invalid verifier name")
#                     sys.exit()
#                 size_verification_test = tester.get_size_verification_test()
#                 print("number of questions", size_verification_test)
#                 #checck optimal
#                 verified = tester.is_agent_value_aligned(opt_policy, Qopt, true_weights)

#                 #print(verified)
#                 if not verified:
#                     print("testing true policy")
                
#                     print("supposed to verify the optimal policy. This is not right!")
#                     input()

#                 correct = 0
#                 for i in range(num_eval_policies):
                    
#                     if debug:
#                         print("\ntesting agent", i)
#                         print("with reward weights:", eval_weights[i])
#                         print("agent policy")
#                         world.print_map(world.to_arrows(eval_policies[i]))
#                         print("compared to ")
#                         print("optimal policy")
#                         true_world.print_map(true_world.to_arrows(opt_policy))
#                         print("true reward weights:", true_weights)
#                         print("mdp features")
#                         utils.display_onehot_state_features(true_world)
#                     verified = tester.is_agent_value_aligned(eval_policies[i], eval_Qvalues[i], eval_weights[i])
#                     #print(verified)
#                     if verified:
#                         if debug:
#                             print("not supposed to be true...")
#                             input()
#                     if not verified:
#                         correct += 1
#                 #TODO: how do I keep track of accuracy??
#                 verifier_accuracy = correct / num_eval_policies
#                 print(verifier_name)
#                 print("Accuracy = ", 100.0*verifier_accuracy)
#                 #input()
                
#                 result_writers[vindx].write("{},{},{}\n".format(correct, num_eval_policies, size_verification_test))
#         for writer in result_writers:
#             writer.close()

#     #teacher = machine_teaching.RankingTeacher(world, debug=False)
#     #teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)