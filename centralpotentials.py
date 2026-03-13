# @2026 Aine Productions.
# Aine Productions is not responsible for shenanigans resulting from the wrongful acquisition, appropriation or misuse of this program.

# File designed to execute an algorithm that creates a 3D central potential for both Yukaw

import numpy as np
import scipy
from matplotlib import pyplot as plt
h = 1
w = 1

mu = 0
ell = 0
a = 50
n_max = 200

# Convert existing potential into one useful for central potentials
def eigenvals(V, h, w):
    
    return 0

# Then compare with existing Coulomb-Yukawa potential
l2 = []
k1 = []

for n in range(0, 2*n_max):
    l2.append(scipy.integrate.quad(lambda x: (1-np.cos(n*np.pi*x)) / (x**2), 0, 1))
    k1.append(scipy.integrate.quad(lambda x: ( 1-np.cos(n*np.pi*x) * np.exp(-1*mu*a*x) ) / x, 0, 1))



# Compare with existing values


