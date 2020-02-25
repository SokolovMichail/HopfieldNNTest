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
        self.w_i_part = partial(weight_initialize,alpha,beta,eks,gamma)
        b_i_part = np.vectorize(partial(bias_initialize,alpha,self.R_matr))
        self.weights = np.zeros(self.dimensions_weights)
        self.weights_init()
        self.bias = np.fromfunction(b_i_part,dimensions)
        self.internal_neuron = (np.random.rand(dimensions[0],dimensions[1],dimensions[2],dimensions[3])*2)-1
        self.output_neurons = (np.vectorize(g))(self.internal_neuron)

    def weights_init(self):
        for i in range(self.dimensions[0]):
            print(1)
            for j in range(self.dimensions[1]):
                for k in range(self.dimensions[2]):
                    for t in range(self.dimensions[3]):
                        for i1 in range(self.dimensions[0]):
                            for j1 in range(self.dimensions[1]):
                                for k1 in range(self.dimensions[2]):
                                    for t1 in range(self.dimensions[3]):
                                        self.weights[i,j,k,t,i1,j1,k1,t1] = self.w_i_part(i,j,k,t,i1,j1,k1,t1)

    def run(self,descents,iterations,delta_t):
        for descent in range(descents):
            for iteration in range(iterations):
                old_output_neurons = (np.vectorize(g))(self.internal_neuron)
                self.step(delta_t)
                self.output_neurons = (np.vectorize(g))(self.internal_neuron)
                stab = self.calculate_stability(old_output_neurons)
                if (stab <= 4):
                    return
        self.random_flip()
        self.renormalize()

    def calculate_stability(self,old_neurons):
        sum = 0
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                for k in range(self.dimensions[2]):
                    for t in range(self.dimensions[3]):
                        if old_neurons[i,j,k,t] != self.output_neurons[i,j,k,t]:
                            sum +=1
        return sum

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
        if random.randint(0,1) == 0:
            self.internal_neuron[i,j,k,t] = 0
        else:
            self.internal_neuron[i,j,k,t] = 1

    def renormalize(self):
        self.internal_neuron = np.vectorize(g)(self.internal_neuron)

    def print_result(self):
        a =  np.nonzero(self.output_neurons)
        i_axis = a[0]
        j_axis = a[1]
        k_axis = a[2]
        t_axis = a[3]
        for i in range(len(a[0])):
            print("[ {0};{1};{2};{3} ]".format(i_axis[i],j_axis[i],k_axis[i],t_axis[i]))
        print(len(a))
        print(self.dimensions)


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
now = time.time()
network = HNN(dimensions,rm,0.2,0.1,0.1,0.1)
network.run(4,10,0.1)
after = time.time()
print(after - now)
network.print_result()









