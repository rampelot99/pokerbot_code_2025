from dist import DDist, uniform_dist, delta_dist, mixture_dist
from util import argmax_with_val, argmax, argmax_choose
import numpy as np

class MDP:
    # states: list or set of states
    # actions: list or set of actions
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward
    # discount_factor: real, greater than 0, less than or equal to 1
    # start_dist: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    #
    # By default, the mdp.policy is greedy; this can be set/defined to something 
    # else after mdp creation, or in subclass
    def __init__(self, states, actions, transition_model, reward_fn,
                 discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)
        self.draw_state = None  # method to display state
        self.q: TabularQ = None

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s, a) and new state, drawn from transition_model.
    #
    # If a terminal state is encountered, get next state from initial
    # state distribution.
    def sim_transition(self, s, a):
        r = self.reward_fn(s, a)
        s_prime = self.transition_model(s, a).draw() \
            if not self.terminal(s) else self.init_state()
        return r, s_prime

    # Return one-hot encoding of state s; used in neural network agent implementations
    def state2vec(self, s):
        v = np.zeros((1, len(self.states)))
        v[0, self.states.index(s)] = 1.0
        return v


class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])

    def get(self, s, a):
        return self.q[(s, a)]

    def set(self, s, a, v):
        self.q[(s, a)] = v

    def update(self, data, lr):
        for (s, a, t) in data:
            old_value = self.get(s, a)
            new_value = (1 - lr) * old_value + lr * t
            self.set(s, a, new_value)

    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy

    def keys(self):
        return self.q.keys()
        
    def __repr__(self):
        return f"TabularQ({repr(self.q)})"


# Simulate an episode (sequence of transitions) of at most
# episode_length, using policy function to select actions.  If we
# find a terminal state, end the episode.  
# 
#  Returns:
#    * accumulated reward
#    * a list of (s, a, r, s') where s' is None for transition from terminal state.
#    * an animation if draw==True, or None if draw==False
def sim_episode(mdp: MDP, policy, episode_length, draw=False, animate=None):
    episode = []
    reward = 0
    s = mdp.init_state()
    all_states = [s]
    for i in range(episode_length):
        a = policy(s)
        (r, s_prime) = mdp.sim_transition(s, a)
        reward += r
        if mdp.terminal(s):
            episode.append((s, a, r, None))
            break
        episode.append((s, a, r, s_prime))
        if draw and mdp.draw_state:
            mdp.draw_state(s)
        s = s_prime
        all_states.append(s)
    if draw:
        print("Creating animation: will appear below...", flush=True)
        animation = animate(all_states, mdp.n, episode_length)
    else:
        animation = None
    return reward, episode, animation

# Return average reward for n_episodes of length episode_length
# while following policy (a function of state) to choose actions.
def evaluate(mdp, policy, n_episodes, episode_length):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e, _ = sim_episode(mdp, policy, episode_length)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score / n_episodes, length / n_episodes

def max_norm_dist(q1: TabularQ, q2: TabularQ):
    u = list(set(q1.keys()) | set(q2.keys()))
    return argmax_with_val(u, lambda sa: abs(q1.get(sa[0], sa[1]) - q2.get(sa[0], sa[1])))[1]


