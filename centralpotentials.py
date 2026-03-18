# @2026 Aine Productions.
# Aine Productions is not responsible for shenanigans resulting from the wrongful acquisition, 
# appropriation or otherwise unintended use of this program.

# File designed to execute an algorithm that solves

import numpy as np
from scipy import integrate, linalg
from matplotlib import pyplot as plt

# Define all SI-constants here.
# This program does not use natural units, for ease of comparison.
hbar = 1.054e-34
m0 = 9.109e-31 # electron mass in kg
e = 1.602e-19 # electron charge in C
eps_0 = 8.85e-12

# Define variables here
ell = 0
a = 450
n_max = 1500


# Define centrifugal potential
def E_cen(ell, r): 
    return ( ell*(ell+1) * hbar**2 ) / (2*m0*r**2)

# Determine eigenvalues from a defined potential (omega(n^3), O likely of higher order)
def eigenvals(V, ell, a, n_max=200):
    # Implement inf-square well energies (E = (pi^2*hbar^2*n^2) / 2*m0*a^2)
    E_0 = lambda n: ((np.pi * hbar * n / a) ** 2 ) / (2 * m0)

    # then define the h-matrix (H) and temporary integration sum (integrand)
    h = np.zeros((n_max, n_max))
    integrand = [0, 0]
    for n in range(0, n_max):
        for m in range(0, n_max):
            integrand = integrate.quad(lambda r: np.sin((n*np.pi*r)/a) * (V(r) + E_cen(ell, a)) * np.sin((m*np.pi*r)/a), 0, a)
            # integrate.quad returns a list (1D array) containing the result itself and the error: grab only the first element
            h[n][m] = np.kron(n, m) * E_0(n) + integrand[0]
    
    # Use acquired h-matrix to determine eigenvalues and -vectors (respectively E and c_n)
    E, phi = linalg.eig(h)
    return E.real, phi


# termine energies of Hydrogen atom potential
V_h = lambda r: -e**2 / (4*np.pi*eps_0*r)
E_hatom, phi_hatom = eigenvals(V_h, ell, a, n_max)
#for n in range(len(E_hatom)):
    #E_hatom[n] /= e # convert from joules to eV

n_hatom = range(0, n_max)
# Compare to Rydberg energies defined in J ( E = -13.6/n^2 * 1.602e-19 ) by plotting both it and the acquired 
E_rydberg = [-13.6 / n**2 * e for n in range(1, n_max)]


print(E_hatom[1], E_hatom[49])
print(E_rydberg[1], E_rydberg[49])
print(E_hatom[1] / E_hatom[49])
print(E_rydberg[1] / E_rydberg[49])

plt.plot(n_hatom, E_hatom)
plt.plot(range(1, n_max), E_rydberg)
plt.xlim(25, 50)
plt.show()
