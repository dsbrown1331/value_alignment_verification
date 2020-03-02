import mdp
import utils
import numpy as np
from alignment_interface import Verifier

class CriticalStateActionValueVerifier(Verifier):
    def __init__(self, mdp_world, critical_threshold, precision = 0.0001, debug=False):
        self.mdp_world = mdp_world
        self.critical_threshold = critical_threshold
        self.precision = precision
        self.debug = debug
        self.q_values = mdp.compute_q_values(mdp_world)
        self.optimal_policy = mdp.find_optimal_policy(mdp_world, Q=self.q_values)

        #find critical states
        self.critical_state_actions = []
        for s in self.mdp_world.states:
            #get best qvalue
            best_action = utils.argmax(self.mdp_world.actions(s), lambda a: self.q_values[s,a])
            average_qvalue = np.mean([self.q_values[s,a] for a in self.mdp_world.actions(s)])
            if  self.q_values[s, best_action] - average_qvalue > self.critical_threshold:
                self.critical_state_actions.append((s,best_action))
        print("Number of critical states", len(self.critical_state_actions))

    def is_agent_value_aligned(self, agent_policy, agent_qvals, agent_reward_wts):


        #Need to ask the agent what it would do in each setting. Just need access to agent's policy
        for s,a in self.critical_state_actions:
            if self.debug:
                print("Testing critical state: ({}, {})".format(s, self.mdp_world.to_arrow(a)))
                print("policy", agent_policy)
                print(type(agent_policy[s]))
            if type(agent_policy[s]) is list: #stochastic optimal policy
                if a not in agent_policy[s]:
                    if self.debug:
                        print("Critical action is not an optimal action for the agent")
                    return False
            else:
                #just a deterministic policy
                if a != agent_policy[s]:
                    if self.debug:
                        print("Critical action is not the optimal action for the agent")
                    return False
        
            if self.debug:
                print("correct answer")
        return True

class CriticalStateEntropyVerifier(Verifier):
    """Based on paper from Anca's group. Find states where entropy is below some handtuned threshold
    and check evaluatoin policy on those states.
    """
    def __init__(self, mdp_world, critical_threshold, precision = 0.0001, debug=False):
        self.mdp_world = mdp_world
        self.entropy_threshold = critical_threshold
        self.precision = precision
        self.debug = debug
        self.q_values = mdp.compute_q_values(mdp_world)
        self.optimal_policy = mdp.find_optimal_policy(mdp_world, Q=self.q_values)

        #find critical states
        if debug:
            print("finding critical states")
        self.critical_state_actions = []
        for s in self.mdp_world.states:
            if debug:
                print(s)
            #calculate entropy of optimal policy (assumes it is stochastic optimal)
            num_optimal_actions = len(self.optimal_policy[s])
            action_probs = np.zeros(len(self.mdp_world.actions(s)))
            for i in range(num_optimal_actions):
                action_probs[i] = 1.0 / num_optimal_actions
            entropy = utils.entropy(action_probs)
            if debug:
                print(s, entropy)
            best_action = utils.argmax(self.mdp_world.actions(s), lambda a: self.q_values[s,a])
            if  entropy < self.entropy_threshold:
                self.critical_state_actions.append((s, best_action))
        

    def is_agent_value_aligned(self, agent_policy, agent_qvals, agent_reward_wts):


        #Need to ask the agent what it would do in each setting. Just need access to agent's policy
        for s,a in self.critical_state_actions:
            if self.debug:
                print("Testing critical state: ({}, {})".format(s, self.mdp_world.to_arrow(a)))
                print("policy", agent_policy)
                print(type(agent_policy[s]))
            if type(agent_policy[s]) is list: #stochastic optimal policy
                if a not in agent_policy[s]:
                    if self.debug:
                        print("Critical action is not an optimal action for the agent")
                    return False
            else:
                #just a deterministic policy
                if a != agent_policy[s]:
                    if self.debug:
                        print("Critical action is not the optimal action for the agent")
                    return False
        
            if self.debug:
                print("correct answer")
        return True