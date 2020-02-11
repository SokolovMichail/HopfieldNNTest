import numpy as np

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


class HNN:
    #Dimensions = c,v,t,p
    #R-Matr = self-explanatory
    def weight_initialize(self,i,j,k,l,i_s,j_s,k_s,l_s):
        return ((-self.alpha)
                * kroneker_delta(i,i_s)
                * kroneker_delta(j,j_s)
                * kroneker_delta(k,k_s)) - \
               (self.beta *
                kroneker_delta(i,i_s) *
                (1 - kroneker_delta(j,j_s)) *
                (1-kroneker_delta(k,k_s) *
                 kroneker_delta(l,l_s))) - \
               (self.eks *
                (1-kroneker_delta(i,i_s))
                * kroneker_delta(j,j_s)
                * (1-kroneker_delta(k,k_s)
                * kroneker_delta(l,l_s))) - \
               (self.gamma *
                (1-kroneker_delta(i,i_s)) *
                (1-kroneker_delta(j,j_s) *
                 kroneker_delta(k,k_s) *
                 kroneker_delta(l,l_s)))

    def bias_initialize(self,i,j,k,l):
        return self.alpha * self.R_matr[i,j,k]

    def __init__(self,dimensions,R_matr,alpha,beta,eks,gamma):
        self.dimensions = dimensions
        self.dimensions_weights = dimensions + dimensions
        self.weights = np.fromfunction(self.weight_initialize,self.dimensions_weights)
        self.bias = np.fromfunction(self.bias_initialize,self.dimensions)
        self.internal_neuron = np.random.rand(dimensions)
        self.output_neurons = g(self.internal_neuron)
        self.R_matr = R_matr
        self.alpha = alpha
        self.beta = beta
        self.eks = eks
        self.gamma = gamma








