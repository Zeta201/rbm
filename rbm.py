import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
#         calculating p(h=1|v)
        p_h_given_v = torch.sigmoid(activation)
#     if we have 100 hidden neurons then p_h_given_v also 100 sized vector
#     this is also the probability that if each neuron i get activated
#     we use these probabilites we will activate each hidden neuron using
#     bernoulli distribution (if p=0.7 then 1 else 0)
#     visible neurons corresponds to ratings (observation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
#         calculating p(v=1|h)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

# contranstive divergence function
# cd => approximating log-likelihood gradient
# since rbm is an energy based model and energy depends on the
# weights of the model (self.W). So we need to optimize the weights
# to minimize the energy. It is also a probabilistic model so minimizing energy
# means maximize the log-likelihood. the algorithm we use to do that is CD.
# This is associated with gibbs sampling. Gibbs consists of creating a gibbs chain in
# K steps by sampling the hidden and visible nodes K times.
# We start with input vector v0 then based on probabilities of p(h|v0) we sample
# the first hidden nodes then we take these hidden nodes (h1) and sample visible nodes
# which is p(v|h1). We do this for k times
# v0 => containing ratings of all the movies by user 1
# vk => visible nodes after k iterations
# ph0 => vector of probabilities p(h=1|v0)
# phk => vector of probabilities p(h=1|vk)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
