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

#evaluate several different verification methods and compute accuracies


def random_weights(num_features):
    rand_n = np.random.randn(num_features)
    l2_ball_weights = rand_n / np.linalg.norm(rand_n)
    return l2_ball_weights
    #return 1.0 - 2.0 * np.random.rand(num_features)

init_seed = 1234
num_trials = 50  #number of mdps with random rewards to try
debug = False
num_rows_list = [4,8,16]
num_cols_list = [4,8,16]
num_features_list = [2,3,4,5,6,7,8]
verifier_list = ["state-value-critical-1.0","state-value-critical-0.5","state-value-critical-0.1", "ranking-halfspace"]
exp_data_dir = "./experiment_data/"


for num_features in num_features_list:
    for num_rows in num_rows_list:
        num_cols = num_rows #keep it square grid for  now

        result_writers = []
        for i, verifier_name in enumerate(verifier_list):
            filename = "{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
            print("writing to", filename)
            result_writers.append(open(exp_data_dir + filename,'w'))

        for r_iter in range(num_trials):
            print("="*10, r_iter, "="*10)
            print("features", num_features, "num_rows", num_rows)
            ##For this test I want to verify that the ranking-based machine teaching is able to correctly verify whether an agent is value aligned or not.
            #MDP is deterministic with fixed number or rows, cols, and features
            #try a variable number of eval policies since bigger domains can have more possible policies (this is just a heuristic to make sure we try a lot but not as many for really small mdps)
            num_eval_policies_tries = 50# 2 * num_features * num_rows * num_cols #Note this isn't how many we'll actually end up with since we reject if same as optimal policy
            initials = [(i,j) for i in range(num_rows) for j in range(num_cols)]
            terminals = []#[(num_rows-1,num_cols-1)]
            gamma = 0.9
            seed = init_seed + r_iter 
            if debug:
                print("seed", seed)
            np.random.seed(seed)
            random.seed(seed)

            #First let's generate a random MDP
            state_features = eutils.create_random_features_row_col_m(num_rows, num_cols, num_features)
            #print("state features\n",state_features)
            true_weights = random_weights(num_features)
            true_world = mdp.LinearFeatureGridWorld(state_features, true_weights, initials, terminals, gamma)
            V = mdp.value_iteration(true_world)
            Qopt = mdp.compute_q_values(true_world, V=V)
            opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt)
            
            if debug:
                print("true weights: ", true_weights)  
            
                print("rewards")
                true_world.print_rewards()
                print("value function")
            
                true_world.print_map(V)
                print("mdp features")
                utils.display_onehot_state_features(true_world)
            
                print("optimal policy")
                true_world.print_map(true_world.to_arrows(opt_policy))
            
            #now find a bunch of other optimal policies for the same MDP but with different weight vectors.
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

            print("There are {} distinct optimal policies".format(len(eval_policies)))
            if len(eval_policies) == 0:
                print("The only possible policy is the optimal policy. There must be a problem with the features. Can't do verification if only on policy possible!")
                sys.exit()
                

            print()
            print("Generating verification tests")

            #TODO: run through all the verifiers and create tests for current MDP
            #TODO: develop a common interface that they all implement 

            
            #TODO: have a list of names
            for vindx, verifier_name in enumerate(verifier_list):
                tester = None
                size_verification_test = None

                if verifier_name == "state-value-critical-1.0":
                    critical_value_thresh = 1.0
                    tester = ah.CriticalStateActionValueVerifier(true_world, critical_value_thresh)
                    size_verification_test = tester.get_size_verification_test()
                elif verifier_name == "state-value-critical-0.5":
                    critical_value_thresh = 0.5
                    tester = ah.CriticalStateActionValueVerifier(true_world, critical_value_thresh)
                    size_verification_test = tester.get_size_verification_test()
                elif verifier_name == "state-value-critical-0.1":
                    critical_value_thresh = 0.1
                    tester = ah.CriticalStateActionValueVerifier(true_world, critical_value_thresh)
                    size_verification_test = tester.get_size_verification_test()
                elif verifier_name == "ranking-halfspace":
                    tester = vav.HalfspaceVerificationTester(true_world, debug = debug)
                    size_verification_test = tester.get_size_verification_test()
                elif verifier_name == "state-optimal-action-ranker":
                    tester = vav.RankingBasedTester(true_world, debug=debug)
                    size_verification_test = test.get_size_verification_test()
                else:
                    print("invalid verifier name")
                    sys.exit()
                #checck optimal
                verified = tester.is_agent_value_aligned(opt_policy, Qopt, true_weights)

                #print(verified)
                if not verified:
                    print("testing true policy")
                
                    print("supposed to verify the optimal policy. This is not right!")
                    input()

                correct = 0
                for i in range(num_eval_policies):
                    
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
                    verified = tester.is_agent_value_aligned(eval_policies[i], eval_Qvalues[i], eval_weights[i])
                    #print(verified)
                    if verified:
                        if debug:
                            print("not supposed to be true...")
                            input()
                    if not verified:
                        correct += 1
                #TODO: how do I keep track of accuracy??
                verifier_accuracy = correct / num_eval_policies
                print(verifier_name)
                print("Accuracy = ", 100.0*verifier_accuracy)
                #input()
                
                result_writers[vindx].write("{},{},{}\n".format(correct, num_eval_policies, size_verification_test))
        for writer in result_writers:
            writer.close()

    #teacher = machine_teaching.RankingTeacher(world, debug=False)
    #teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)