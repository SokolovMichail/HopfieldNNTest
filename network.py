import numpy as np
from functools import partial
import time
import random

def kroneker_delta(i1,i2):
    if (i1 == i2):
        return 1
    else:
        return 0

def g(x):
    if (x > 0):
        return 1
    else:
        return 0

def weight_initialize(alpha,beta,eks,gamma,i,j,k,l,i_s,j_s,k_s,l_s):
        a = ((alpha)
                * kroneker_delta(i,i_s)
                * kroneker_delta(j,j_s)
                * kroneker_delta(k,k_s)) - \
               (beta *
                kroneker_delta(i,i_s) *
                (1 - kroneker_delta(j,j_s)) *
                (1-kroneker_delta(k,k_s) *
                 kroneker_delta(l,l_s))) - \
               (eks *
                (1-kroneker_delta(i,i_s))
                * kroneker_delta(j,j_s)
                * (1-kroneker_delta(k,k_s)
                * kroneker_delta(l,l_s))) - \
               (gamma *
                (1-kroneker_delta(i,i_s)) *
                (1-kroneker_delta(j,j_s) *
                 kroneker_delta(k,k_s) *
                 kroneker_delta(l,l_s)))
        return a

def bias_initialize(alpha,R_matr,i,j,k,l):
    i = int(i)
    j = int(j)
    k = int(k)
    l = int(l)
    elem = R_matr[i,j,k]
    return alpha * elem

class HNN:
    #Dimensions = c,v,t,p
    #R-Matr = self-explanatory
    def __init__(self,dimensions,R_matr,alpha,beta,eks,gamma):
        self.dimensions = dimensions
        self.dimensions_weights = dimensions + dimensions
        self.R_matr = R_matr
        self.alpha = alpha
        self.beta = beta
        self.eks = eks
        self.gamma = gamma
        w_i_part = np.vectorize(partial(weight_initialize,alpha,beta,eks,gamma))
        b_i_part = np.vectorize(partial(bias_initialize,alpha,self.R_matr))
        self.weights = np.fromfunction(w_i_part, self.dimensions_weights)
        self.bias = np.fromfunction(b_i_part,dimensions)
        self.internal_neuron = np.random.rand(dimensions[0],dimensions[1],dimensions[2],dimensions[3])
        self.output_neurons = (np.vectorize(g))(self.internal_neuron)

    def run(self,descents,iterations,delta_t):
        for descent in range(descents):
            for iteration in range(iterations):
                self.step(delta_t)
                stab = np.sum(self.output_neurons)
                if (stab > 1):
                    return
        self.random_flip()
        self.renormalize()


    def step(self,delta_t):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                for k in range(self.dimensions[2]):
                    for t in range(self.dimensions[3]):
                        self.internal_neuron[i,j,k,t] += delta_t * (np.sum(self.weights,(4,5,6,7))[i,j,k,t] + self.bias[i,j,k,t])


    def random_flip(self):
        i = random.randint(0,self.dimensions[0])
        j = random.randint(0, self.dimensions[1])
        k = random.randint(0, self.dimensions[2])
        t = random.randint(0, self.dimensions[3])
        self.internal_neuron[i,j,k,t]*=-1

    def renormalize(self):
        self.internal_neuron = np.vectorize(g)(self.internal_neuron)


dimensions = (2,11,10,24)
rm = np.zeros((2,11,10))
#7 gr
rm[0,0,1] = 3
rm[0,4,1] = 1
rm[0,5,4] = 2
rm[0,8,1] = 2
rm[0,1,0] = 1
rm[0,2,7] = 1
rm[0,10,9] = 1
rm[0,10,8] = 1
rm[0,2,0] = 1
#8gr
rm[1,2,0] = 1
rm[1,0,1] = 2
rm[1,1,2] = 1
rm[1,4,1] = 1
rm[1,6,5] = 2
rm[1,8,1] = 2
rm[1,2,7] = 1
rm[1,10,9] = 1
rm[1,9,7] = 1
rm[1,10,7] = 1
# #9gr
# rm[2,2,1] = 1
# rm[2,0,1] = 1
# rm[2,3,3] = 1
# rm[2,0,1] = 1
# rm[2,4,1] = 1
# rm[2,7,6] = 1
# rm[2,8,1] = 2
# rm[2,2,7] = 1
# rm[2,10,9] = 1
# rm[2,7,6] = 1
# rm[2,2,8] =1

network = HNN(dimensions,rm,0.2,0.1,0.1,0.1)
now = time.time()
network.run(2,5,0.1)
after = time.time()
print(after - now)
print(network.output_neurons)







