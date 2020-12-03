#I want to rerun things with the eps-ARP rather than the AEC and see if I can break it

import sys
import os
exp_path = os.path.dirname(os.path.abspath(__file__))
print(exp_path)
project_path = os.path.abspath(os.path.join(exp_path, "..", ".."))
sys.path.insert(0, project_path)
print(sys.path)

import src.experiment_utils as eutils
import src.utils as utils
import src.mdp as mdp
import src.machine_teaching
import copy
import numpy as np
import src.value_alignment_verification as vav
import src.alignment_heuristics as ah
import random
import sys

#evaluate several different verification methods and compute accuracies


def random_weights(num_features):
    rand_n = np.random.randn(num_features)
    l2_ball_weights = rand_n / np.linalg.norm(rand_n)
    return l2_ball_weights
    #return 1.0 - 2.0 * np.random.rand(num_features)

init_seed = 1234
num_trials = 20  #number of mdps with random rewards to try
num_eval_policies_tries = 50

eps_gap = 0.0  #how suboptimal can things be before we reject as misaligned

#scot params
num_rollouts = 20
#used for scot and traj comparisons
rollout_length = 30  #should be less than  np.log(eps * (1-gamma))/np.log(gamma) to gurantee epsilong accuracy


debug = False
precision = 0.00001
num_rows_list = [3]
num_cols_list = [3]
num_features_list = [3]
#verifier_list = ['trajectory_aec',"ranking-halfspace","scot", "optimal_action","state-value-critical-1.0", "state-value-critical-0.7","state-value-critical-0.5","state-value-critical-0.2","state-value-critical-0.1","state-value-critical-0.01"]
verifier_list = ['ranking-halfspace']

exp_data_dir = os.path.join(project_path, "results", "arp_version")

if not os.path.exists(exp_data_dir):
    os.makedirs(exp_data_dir)

for num_features in num_features_list:
    for num_rows in num_rows_list:
        num_cols = num_rows #keep it square grid for  now

        result_writers = []
        for i, verifier_name in enumerate(verifier_list):
            filename = "arp{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
            full_path = os.path.join(exp_data_dir, filename)
            print("writing to", full_path)
            result_writers.append(open(full_path,'w'))
            #input()

        for r_iter in range(num_trials):
            print("="*10, r_iter, "="*10)
            print("features", num_features, "num_rows", num_rows)
            ##For this test I want to verify that the ranking-based machine teaching is able to correctly verify whether an agent is value aligned or not.
            #MDP is deterministic with fixed number or rows, cols, and features
            #try a variable number of eval policies since bigger domains can have more possible policies (this is just a heuristic to make sure we try a lot but not as many for really small mdps)
            # 2 * num_features * num_rows * num_cols #Note this isn't how many we'll actually end up with since we reject if same as optimal policy
            initials = [(i,j) for i in range(num_rows) for j in range(num_cols)]
            terminals = []#[(num_rows-1,num_cols-1)]
            gamma = 0.9
            seed = 1240#init_seed + r_iter 
            print("seed", seed)
            np.random.seed(seed)
            random.seed(seed)

            #First let's generate a random MDP
            state_features = eutils.create_random_features_row_col_m(num_rows, num_cols, num_features)
            #print("state features\n",state_features)
            true_weights = random_weights(num_features)
            true_world = mdp.LinearFeatureGridWorld(state_features, true_weights, initials, terminals, gamma)
            V = mdp.value_iteration(true_world, epsilon=precision)
            true_exp_return = np.mean([V[s] for s in true_world.initials])
            Qopt = mdp.compute_q_values(true_world, V=V, eps=precision)
            opt_policy = mdp.find_optimal_policy(true_world, Q = Qopt, epsilon=precision)
            
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

                print("Qvals")
                for s in true_world.states:
                    print()
                    for a in true_world.actlist:
                        print("Q({},{}) = {}".format(s,true_world.to_arrow(a), Qopt[s,a]))
            
            #now find a bunch of other optimal policies for the same MDP but with different weight vectors.
            world = copy.deepcopy(true_world)
            eval_policies = []
            eval_Qvalues = []
            eval_weights = []
            eval_policy_losses = []
            num_eval_policies = 0
            for i in range(num_eval_policies_tries):
                #print("trying", i)
                #change the reward weights
                eval_weight_vector = random_weights(num_features)
                #TODO maybe make above random normal centered on truth
                world.weights = eval_weight_vector
                #print(eval_weight_vector)
                #find the optimal policy under this MDP
                Veval = mdp.value_iteration(world, epsilon=precision)
                Qval = mdp.compute_q_values(world, V=Veval, eps=precision)
                eval_policy = mdp.find_optimal_policy(world, Q=Qval, epsilon=precision)
                #world.print_map(world.to_arrows(eval_policy))
                eval_exp_values = mdp.policy_evaluation(eval_policy, true_world, epsilon=precision)
                eval_exp_return = np.mean([eval_exp_values[s] for s in true_world.initials])
                #print("true exp return", true_exp_return, "eval exp return", eval_exp_return)
                eval_policy_loss = true_exp_return - eval_exp_return
                #print("diff", eval_policy_loss)
                #only save if not equal to optimal policy
                if eval_policy != opt_policy and eval_policy not in eval_policies:
                    if debug:
                        print("found distinct eval policy")
                        world.print_map(world.to_arrows(eval_policy))
                
                    eval_policies.append(eval_policy)
                    eval_Qvalues.append(Qval)
                    eval_weights.append(eval_weight_vector)
                    eval_policy_losses.append(eval_policy_loss)
                    num_eval_policies += 1

            print("There are {} distinct optimal policies".format(len(eval_policies)))
            if len(eval_policies) == 0:
                print("The only possible policy is the optimal policy. There must be a problem with the features. Can't do verification if only one policy possible!")
                sys.exit()
                

            print()
            print("Generating verification tests")

            #TODO: run through all the verifiers and create tests for current MDP
            #TODO: develop a common interface that they all implement 

            
            #TODO: have a list of names
            for vindx, verifier_name in enumerate(verifier_list):
                tester = None
                size_verification_test = None

                if "state-value-critical-" in verifier_name:
                    critical_value_thresh = float(verifier_name[len("state-value-critical-"):])
                    #print("critical value", critical_value_thresh)
                    tester = ah.CriticalStateActionValueVerifier(true_world, critical_value_thresh, precision=precision, debug=debug)
                elif verifier_name == "ranking-halfspace":
                    tester = vav.HalfspaceVerificationTester(true_world, debug = debug, precision=precision, epsilon_gap=eps_gap)

                elif verifier_name == "state-optimal-action_ranker":
                    tester = vav.RankingBasedTester(true_world, precision, debug=debug)

                elif verifier_name == "optimal_action":
                    tester = vav.OptimalRankingBasedTester(true_world, precision, debug=debug)
                
                elif verifier_name == "trajectory_aec":
                    tester = vav.TrajectoryRankingBasedTester(true_world, precision, rollout_length, debug=debug)

                elif verifier_name == "optimal_action_allquestions":
                    tester = vav.OptimalRankingBasedTesterAll(true_world, precision, debug=debug)

                elif verifier_name == "scot":
                    tester = vav.SCOTVerificationTester(true_world, precision, num_rollouts, rollout_length, debug=debug)
                
                else:
                    print("invalid verifier name")
                    sys.exit()
                size_verification_test = tester.get_size_verification_test()
                print("number of questions", size_verification_test)
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
                        print("policy loss = ", eval_policy_losses[i])
                        
                        print("true reward weights:", true_weights)
                        print("mdp features")
                        utils.display_onehot_state_features(true_world)
                    verified = tester.is_agent_value_aligned(eval_policies[i], eval_Qvalues[i], eval_weights[i])
                    #print(verified)
                    if verified:
                        if eval_policy_losses[i] > eps_gap:
                            print("verified but shouldn't be based on policy loss", eval_policy_losses[i])
                            input()
                        else:
                            correct += 1
                        if debug:
                            print("not supposed to be true...")
                            input()
                    if not verified:
                        if eval_policy_losses[i] < eps_gap:
                            print("not verified but should be based on policy loss", eval_policy_losses[i])
                        else:
                            correct += 1
                #TODO: how do I keep track of accuracy??
                verifier_accuracy = correct / num_eval_policies
                print(verifier_name)
                print("Accuracy = ", 100.0*verifier_accuracy)
                #input()
                
                #result_writers[vindx].write("{},{},{}\n".format(correct, num_eval_policies, size_verification_test))
        #for writer in result_writers:
        #    writer.close()

    #teacher = machine_teaching.RankingTeacher(world, debug=False)
    #teacher.get_optimal_value_alignment_tests(use_suboptimal_rankings = False)