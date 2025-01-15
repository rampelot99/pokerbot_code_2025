import torch
import torch.nn as nn
import numpy as np

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

def make_nn(state_dim, num_layers, num_units):
    """
    Make a ReLU-activated neural network (i.e., ReLU-activated MLP)
    with the specified details. All hidden layers have the same size of num_units

    state_dim: (int) number of states
    num_layers: (int) number of fully connected hidden layers
    num_units: (int) number of dense relu units to use in hidden layers
    """
    model = []
    model += [nn.Linear(state_dim, num_units), nn.ReLU()]
    for i in range(num_layers - 1):
        model += [nn.Linear(num_units, num_units), nn.ReLU()]
    model += [nn.Linear(num_units, 1)]
    model = nn.Sequential(*model)
    return model