import numpy as np
import utils

#the following code is adapted from  erensezener/aima-based-irl 

class LinearFeatureGridWorld:
    """A Markov Decision Process, defined by an initial state distribution, 
    transition model, reward function, gamma, and action list"""

    def __init__(self, features, weights, initials, terminals, gamma=.95):
        self.features = features
        self.weights = weights
        self.initials=initials
        self.actlist=[(1,0),(-1,0), (0,1), (0,-1)] #down, up, right, left
        self.terminals=terminals
        self.gamma=gamma
        self.rows, self.cols = len(features), len(features[0])
        self.states = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if features[r][c] is not None: #if features are None then there is an obstacle
                    self.states.add((r,c))
        

    def R(self, state):
        "Return a numeric reward for this state."
        r,c = state
        return np.dot(self.features[r][c], self.weights)

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(1.0, self.go(state, action))]

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a 
        fixed list of actions, except for terminal states. Override this 
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = self.vector_add(state, direction)
        if state1 in self.states: #move if a valid next state (i.e., not off the grid or into obstacle)
            return state1
        else: #self transition
            return state

    def vector_add(self, a, b):
        """Component-wise addition of two vectors.
        >>> vector_add((0, 1), (8, 9))
        (8, 10)
        """
        return a[0] + b[0], a[1] + b[1]


    def to_arrows(self, policy):
        chars = {(1, 0): 'v', (0, 1): '>', (-1, 0): '^', (0, -1): '<', None: '.'}
        policy_arrows = {}
        for (s,a_list) in policy.items():
            #concatenate optimal actions
            opt_actions = ""
            for a in a_list:
                opt_actions+=chars[a]
            policy_arrows[s] = opt_actions
        return policy_arrows

    def print_policy(self, policy):
        arrow_map = self.to_arrows(policy)


    def to_grid(self, mapping):
        """Convert a mapping from (r, c) to val into a [[..., val, ...]] grid."""
        return list([[mapping.get((r, c), None)
                               for c in range(self.cols)]
                              for r in range(self.rows)])

    def print_2darray(self, array_2d):
        """take a 2-d array of values and print nicely"""
        for r in (range(self.rows)):
            for c in (range(self.cols)):
                if type(array_2d[r][c]) is float:
                    print("{:0.4f}".format(array_2d[r][c], 3), end="\t")
                else:
                    print("{}".format(array_2d[r][c], 3), end="\t")
            print()

    def print_map(self, mapping):
        array2d = self.to_grid(mapping)
        self.print_2darray(array2d)


    def print_rewards(self):
        for x in (range(self.rows)):
            for y in (range(self.cols)):
                print("{:0.4f}".format(self.R((x,y))), end="\t")
            print()

    def get_grid_size(self):
        return len(self.grid), len(self.grid[0])


#______________________________________________________________________________




def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration."
    V1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        V = V1.copy()
        delta = 0
        for s in mdp.states:
            V1[s] = R(s) + gamma * max([sum([p * V[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(V1[s] - V[s]))
        #print V1
        if delta < epsilon * (1 - gamma) / gamma:
            return V


def compute_q_values(mdp, V=None):
    if not V:
        #first we need to compute the value function
        V = value_iteration(mdp)
    Q = {}
    for s in mdp.states:
        for a in mdp.actions(s):
            Qtemp = mdp.R(s)
            for (p, sp) in mdp.T(s, a):
                Qtemp += mdp.gamma * p * V[sp]
            Q[s, a] = Qtemp
    return Q



def find_optimal_policy(mdp, V=None, Q=None):
    """Given an MDP and an optional value function V or optional Q-value function, determine the best policy,
    as a mapping from state to action."""
    #check if we need to compute Q-values
    if not Q:
        if not V:
            Q = compute_q_values(mdp)
        else:
            Q = compute_q_values(mdp, V)

    pi = {}
    for s in mdp.states:
        #find all optimal actions

        pi[s] = utils.argmax_list(mdp.actions(s), lambda a: Q[s,a])
    return pi


def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


#______________________________________________________________________________


#TODO what is a good value for k?
def policy_evaluation(pi, U, mdp, k=100):
    """Return an updated utility mapping U from each state in the MDP to its 
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        #print "k", i
        for s in mdp.states:
            #print "s", s
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
            #print "U[s]", U[s]
    return U

#how to generate a demo from a start location
###TODO  this is for deterministic settings!!
###TODO add a terminationg criterion like value and policy iteration!
def generate_demonstration(start, policy, mdp):
    """given a start location return the demonstration following policy
    return a state action pair array"""
    
    demonstration = []
    curr_state = start
    #print('start',curr_state)
    
    while curr_state not in mdp.terminals:
        #print('action',policy[curr_state])
        demonstration.append((curr_state, policy[curr_state]))
        curr_state = mdp.go(curr_state, policy[curr_state])
        #print('next state', curr_state)
    #append the terminal state
    demonstration.append((curr_state, None))
    return demonstration