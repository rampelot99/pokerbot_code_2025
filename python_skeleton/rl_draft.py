import mdp

import importlib
import numpy as np
import torch
import random

from collections.abc import Callable

# Discrete distribution represented as a dictionary.  Can be
# sparse, in the sense that elements that are not explicitly
# contained in the dictionary are assumed to have zero probability.
class DDist:
    def __init__(self, dictionary):
        # Initializes dictionary whose keys are elements of the domain
        # and values are their probabilities
        self.d = dictionary
        self.keys = list(self.d.keys())
        total_probability = sum(self.d.values())
        self.probabilities = [self.d[k]/total_probability for k in self.keys]

    def prob(self, elt):
        # Returns the probability associated with elt
        return self.d[elt]

    def support(self):
        # Returns a list (in any order) of the elements of this
        # distribution with non-zero probabability.
        return [k for k in self.probabilities if k != 0]

    def draw(self):
        # Returns a randomly drawn element from the distribution
        random.choices(self.keys, weights=self.probabilities, k=1)

    def expectation(self, f):
        # Returns the expected value of the function f over the current distribution
        # f is a function that returns a value given an element in the distribution
        return sum([f(k)*self.probabilities[idx] for idx, k in enumerate(self.keys)])

    def get_all_probs(self):
        # Returns a list of (element, probability) tuple for all elements
        # with non-zero probabability
        return [(k, self.probabilities[idx]) for idx, k in enumerate(self.keys) if self.probabilities[idx] != 0]

def uniform_dist(elts):
    # Returns DDist with uniform distribution over a given finite set of elts
    d = {e : 1 for e in elts}
    return DDist(d)

def argmax(l, f):
    # l is a list of items. f is a procedure that maps an item into a numeric score
    # returns the element of l that has the highest score
    scores = {i: f(i) for i in l}
    return max(scores, key=lambda:scores.get)

class MDP:
    def __init__(self, states, actions, transition_model, reward_fn,
                     discount_factor = 1.0, start_dist = None):
        # NOTE: we set all of the following variables as class variables
        # that can be accessed through mdp.variable_name,
        # where mdp is an MDP object, e.g., mdp.states.
        self.states = states
        self.actions = actions
        self.transition_model: Callable[[tuple], DDist] = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.q: TabularQ = None # TODO: add q function
        if start_dist is None:
            self.start_dist = uniform_dist(states)
        else:
            self.start_dist = start_dist
        # states: list or set of states
        # actions: list or set of actions
        # transition_model: function from (state, action) into DDist over next state
        # reward_fn: function from (state, action) to real-valued reward
        # discount_factor: real, greater than 0, less than or equal to 1
        # start_dist: optional instance of DDist, specifying initial state dist
        #    if it's unspecified, we'll use a uniform over states

    def terminal(self, s):
        # Given a state, return True if the state should be considered to
        # be terminal.  You can think of a terminal state as generating an
        # infinite sequence of zero reward.
        pass

    def init_state(self):
        # Return an initial state by drawing from the distribution over start states.
        return self.start_dist.draw()

    def sim_transition(self, s, a):
        # Simulates a transition from the given state, s and action a, using the
        # transition model as a probability distribution.  If s is terminal,
        # uses init_state to draw an initial state.  Returns (reward, new_state)
        if self.terminal(s):
            new_s = self.init_state()
        else:
            d = self.transition_model(s, a)
            new_s = d.draw()
        reward = self.reward_fn(s, a)
        return reward, new_s

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])

    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy

    def set(self, s, a, v):
        self.q[(s,a)] = v

    def get(self, s, a):
        return self.q[(s,a)]
    
    def update(self, data, lr):
        for (s, a, t) in data:
            old_value = self.get(s, a)
            new_value = (1 - lr) * old_value + lr * t
            self.set(s, a, new_value)
    
# Make sure you rerun this if you update your implementations!

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

# Simulate an episode (sequence of transitions) of at most
# episode_length, using policy function to select actions.  If we
# find a terminal state, end the episode.  
# 
#  Returns:
#    * accumulated reward
#    * a list of (s, a, r, s') where s' is None for transition from terminal state.
#    * an animation if draw==True, or None if draw==False
def sim_episode(mdp, policy, episode_length, draw=False, animate=None):
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

def max_norm_dist(q1, q2):
    u = list(set(q1.keys()) | set(q2.keys()))
    return argmax_with_val(u, lambda sa: abs(q1.get(sa[0], sa[1]) - q2.get(sa[0], sa[1])))[1]


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

class NNQ:
    def __init__(self, states, actions, state2vec, num_layers, num_units, lr=1e-2, epochs=1):
        self.running_loss = 0.0  # To keep a running average of the loss
        self.running_one = 0.
        self.num_running = 0.001
        self.states = states
        self.actions = actions
        self.state2vec = state2vec
        self.epochs = epochs
        self.lr = lr
        state_dim = state2vec(states[0]).shape[1] # a row vector
        self.models = {a : make_nn(state_dim, num_layers, num_units) for a in actions}
        # Your code here

    def predict(self, model, s):
      return model(torch.FloatTensor(self.state2vec(s))).detach().numpy()

    def get(self, s, a):
        return self.predict(self.models[a],s)

    def fit(self, model, X,Y, epochs=None, dbg=None):
      if epochs is None:
         epochs = self.epochs
      train = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
      train_loader = torch.utils.data.DataLoader(train, batch_size=256,shuffle=True)
      opt = torch.optim.SGD(model.parameters(), lr=self.lr)
      for epoch in range(epochs):
        for (X,Y) in train_loader:
          opt.zero_grad()
          loss = torch.nn.MSELoss()(model(X), Y)
          loss.backward()
          self.running_loss = self.running_loss*(1.-self.num_running) + loss.item()*self.num_running
          self.running_one = self.running_one*(1.-self.num_running) + self.num_running
          opt.step()
      # if dbg is True or (dbg is None and np.random.rand()< (0.001*X.shape[0])):
      #   print('Loss running average: ', self.running_loss/self.running_one)

    # LOOP THROUGH ALL ACTIONS
    def update(self, data, lr, dbg=None):
        for action in self.actions:
          X = []
          Y = []
          for s, a, t in data:
            if a == action:
              X.append(self.state2vec(s))
              Y.append(t)
          # TRAIN MODEL PER ACTION
          if X and Y:
            X = np.vstack(X)
            Y = np.vstack(Y)
            self.fit(self.models[action], X, Y, epochs = self.epochs, dbg = dbg)
