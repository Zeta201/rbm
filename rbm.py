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

    def fit(self, training_set, nb_users, epochs, batch_size, steps):
        self.training_set = training_set
        self.nb_users = nb_users
        for epoch in range(1, epochs+1):
            train_loss = 0
            s = 0.
            for id_user in range(0, nb_users - batch_size, batch_size):
                vk = self.training_set[id_user:id_user+batch_size]
                v0 = self.training_set[id_user:id_user+batch_size]
                ph0, _ = self.sample_h(v0)
                # contrastive divergence
                for k in range(steps):
                    _, hk = self.sample_h(vk)
                    _, vk = self.sample_v(hk)
                    vk[v0 < 0] = v0[v0 < 0]
                phk, _ = self.sample_h(vk)
                self.train(v0, vk, ph0, phk)
                train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
                s += 1.
            print('ephoch: {} loss: {}'.format(epoch, train_loss/s))

    def predict(self, test_set):
        self.test_set = test_set
        test_loss = 0
        s = 0
        for id_user in range(self.nb_users):
            v = self.training_set[id_user:id_user+1]
            vt = self.test_set[id_user:id_user+1]
            if len(vt[vt >= 0]) > 0:
                _, h = self.sample_h(v)
                _, v = self.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
                s += 1.
        print('test_loss: {}'.format(test_loss/s))
