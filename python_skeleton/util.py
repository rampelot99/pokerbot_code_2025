import numpy as np
import random

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return rv(value_list).T

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

def argmax_choose(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score. If
              more than one have the same highest score, we choose
              one randomly.
    """
    vals = [f(x) for x in l]
    max_val = max(vals)
    max_indices = [i for i, val in enumerate(vals) if val == max_val]
    return l[random.choice(max_indices)]

def argmax_with_val(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score and the score
    """
    best = l[0]
    best_score = f(best)
    for x in l:
        x_score = f(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return (best, best_score)
