import numpy as np

def entropy(outcome_probs):
    entropy = 0.0
    for p in outcome_probs:
        if p != 0:
            entropy -= p * np.log(p)
    return entropy


##utility functions for value alignment and mdps



def display_onehot_state_features(mdp_world):
    state_features = mdp_world.features
    state_features_2d = []
    for r in range(mdp_world.rows):
        row_features = []
        for c in range(mdp_world.cols):
            row_features.append(state_features[r][c].index(1))
        state_features_2d.append(row_features)
    mdp_world.print_2darray(state_features_2d)






def print_question(question, mdp_world):    
    arrow = mdp_world.to_arrow #to make debugging actions human readable
    if len(question) == 2:
        (s,a), (s,b) = question
        print("[{},{}] < [{},{}]".format(s,arrow(a),s,arrow(b)))
    else:
        (s,a), (s,b), equivalent = question
        if equivalent:
            print("[{},{}] = [{},{}]".format(s,arrow(a),s,arrow(b)))
        else:
            print("[{},{}] < [{},{}]".format(s,arrow(a),s,arrow(b)))





#______________________________________________________________________________
# Functions on sequences of numbers
# NOTE: these take the sequence argument first, like min and max,
# and like standard math notation: \sigma (i = 1..n) fn(i)
# A lot of programing is finding the best value that satisfies some condition;
# so there are three versions of argmin/argmax, depending on what you want to
# do with ties: return the first one, return them all, or pick at random.


def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmin_list(seq, fn, precision):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        if x_score < best_score - precision:
            best, best_score = [x], x_score
        elif abs(x_score - best_score) < precision:
            best.append(x)
    return best

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmax_list(seq, fn, precision):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x), precision)

def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x))