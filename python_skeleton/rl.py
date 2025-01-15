from util import argmax
from mdp import TabularQ, MDP, uniform_dist, sim_episode
import random
import engine

def value(q: TabularQ, s):
    return max(q.get(s, a) for a in q.actions)

def greedy(q: TabularQ, s):
    return argmax(q.actions, lambda a: q.get(s, a))

# Note: Different than in previous homework. Here, we expect
# mdp to have mdp.q holding the q function.
# (i.e. Store your updated q in mdp.q rather than returning it.)
def value_iteration(mdp: MDP, eps=0.01, max_iters=1_000):
    def v(s):
        return value(mdp.q,s)
    for it in range(max_iters):
        new_q = mdp.q.copy()
        delta = 0
        for s in mdp.states:
            for a in mdp.actions:
                new_q.set(s, a, mdp.reward_fn(s, a) + mdp.discount_factor * \
                          mdp.transition_model(s, a).expectation(v))
                delta = max(delta, abs(new_q.get(s, a) - mdp.q.get(s, a)))
        if delta < eps:
            return new_q
        mdp.q = new_q
    return mdp.q

def epsilon_greedy(q: TabularQ, s, eps = 0.5):
    """ Returns an action.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:  # True with prob eps, random action
        return uniform_dist(q.actions).draw()
    else:                   # False with prob 1-eps, greedy action
        return greedy(q, s)
    

def Q_learn(mdp: MDP, lr=0.1, iters=100, eps=0.5, interactive_fn=None):
    s = mdp.init_state()
    for i in range(iters):
        a = epsilon_greedy(mdp.q, s, eps)
        r, s_prime = mdp.sim_transition(s, a)
        future_val = 0 if mdp.terminal(s) else value(mdp.q, s_prime)
        mdp.q.update([(s, a, (r + mdp.discount_factor * future_val))], lr)
        s = s_prime
        if interactive_fn:
            interactive_fn(i)

def Q_learn_batch(mdp: MDP, lr=0.1, iters=100, eps=0.5,
                  episode_length=10, n_episodes=2, interactive_fn=None):
    all_experiences = []
    policy = lambda s: epsilon_greedy(mdp.q, s, eps)
    for i in range(iters):
        for e in range(n_episodes):
            _, episode,_ = sim_episode(mdp, policy, episode_length)
            all_experiences += episode
        all_q_targets = []
        for (s, a, r, s_prime) in all_experiences:
            future_val = 0 if s_prime is None else value(mdp.q, s_prime)
            all_q_targets.append((s, a, (r + mdp.discount_factor * future_val)))
        mdp.q.update(all_q_targets, lr)
        if interactive_fn:
            interactive_fn(i)


trans = engine.RoundState().proceed

states = []

def reward(s, a):
    pot_odds = s["continue_cost"] / (s["my_pip"] + s["opp_pip"] + 0.1)
    if a == engine.FoldAction:
        return pot_odds
    return 1-pot_odds

mdp = MDP(states, [engine.FoldAction, engine.CallAction, engine.CheckAction, engine.RaiseAction], trans, reward, 0.9)

