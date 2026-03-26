# @2026 Aine Productions.
# Aine Productions is not responsible for shenanigans resulting from the wrongful acquisition, 
# appropriation or otherwise unintended use of this program.

# File designed to execute an algorithm that solves

import numpy as np
from tqdm import tqdm
from scipy import integrate, linalg
from matplotlib import pyplot as plt

# Define all SI-constants here.
# This program does not use natural units, for ease of comparison.
# hbar = 1.054e-34
# m0 = 9.109e-31 # electron mass in kg
# e = 1.602e-19 # electron charge in C
# eps_0 = 8.85e-12
# r0 = 5.292e-11 # Bohr radius in m
hbar = 1
m0 = 1
e = 1
eps_0 = 1 / (4*np.pi)
r0 = (4 * np.pi * eps_0 * (hbar**2)) / (m0*(e**2)) # Bohr radius

# Define variables here
ell = 0
a = 20 * r0
n_max = 200


# Define centrifugal potential
def E_cen(ell, r): 
    return -1 * ( ell*(ell+1) * hbar**2 ) / (2*m0*r**2)

# Determine eigenvalues from a defined potential (omega(n^3), O likely of higher order)
def eigenvals(V, ell, a, n_max=200):
    # Implement inf-square well energies (E = (pi^2*hbar^2*n^2) / 2*m0*a^2) => (pi*hbar*n/a)^2 / 2*m0
    E_0 = lambda n: ((np.pi * hbar * n / a) ** 2 ) / (2 * m0)

    # then define the h-matrix (H) and temporary integration sum (integrand)
    h = np.zeros((n_max, n_max))
    integrand = [0, 0]
    for n in range(0, n_max):
        for m in range(0, n_max):
            integrand = integrate.quad(lambda r: (2 / a) * np.sin((n*np.pi*r)/a) * (V(r) + E_cen(ell, a)) * np.sin((m*np.pi*r)/a), 0, a)
            # integrate.quad returns a list (1D array) containing the result itself and the error: grab only the first element
            h[n][m] = integrand[0]
            if (n == m):
                h[n][m] += E_0(n)
    
    # Use acquired h-matrix to determine eigenvalues and -vectors (respectively E and c_n)
    E, phi = linalg.eig(h)
    return np.sort(E.real), np.sort(phi)

def hEnergyPlot():
    # Determine energies of Hydrogen atom potential
    # V_h = lambda r: -e**2 / (4*np.pi*eps_0*r)
    V_h = lambda r: -1 / r
    E_hatom, phi_hatom = eigenvals(V_h, ell, a, n_max)
    #for n in range(len(E_hatom)):
        #E_hatom[n] /= e # convert from joules to eV

    n_hatom = range(0, n_max)
    # Compare to Rydberg energies defined in eV ( E = -13.6/n^2 ), but expressed in atomic units (unit of energy is roughly 27.211 eV) 
    # by plotting both it and the acquired energies
    E_rydberg = [(-13.6 / n**2) / 27.211 for n in range(1, n_max)]


    # print(E_hatom[0], E_hatom[49])
    # print(E_rydberg[0], E_rydberg[49])
    # print(E_hatom[0] / E_rydberg[0])
    # print(E_hatom[49] / E_rydberg[49])

    plt.plot(n_hatom, E_hatom, "-")
    plt.plot(range(1, n_max), E_rydberg, "o")
    plt.xlim(0, 25)
    plt.ylim(-1, 5)
    plt.ylabel("energy [27.211 eV]")
    plt.xlabel("n")
    plt.legend(["Program energies", "Rydberg energies"])
    plt.show()

# Second step: determine critical potentials
def criticalPotentials():
    critical_potentials = np.zeros(round(a))
    for b in np.arange(0, (int(a) / 10), 0.01):
        for V_0 in tqdm(np.arange(0.2, 0.3, 0.001)):
            # define finite sphere-well potential mid-loop, so to allow V(r) formatting
            def V_sphere(r):
                if (r > b or r < 0):
                    return 0
                else:
                    return (-1 * V_0)
                
            E_sphere, phi_sphere = eigenvals(V_sphere, ell, a, 100)
            # To test whether bound eigenstates exist, test whether the lowest possible one is unbound or not.
            if (E_sphere[0] > 0):
                critical_potentials[b] = V_0 - 0.001
                break

    length_array = np.arange(0, 0.1, (a**-1))
    plt.plot(length_array, critical_potentials)
    plt.xlabel("b/a")
    plt.ylabel("critical potential V_c")
    plt.plot()

# Step three: determine critical coefficient of Yukawa potential
def yukawaCriticals():
    A = 1
    critical_coefficient = 0
    for mu in range(0.90, 1.5, 0.01):
        V_yukawa = lambda r: (np.e ** (mu * r / r0)) / r
        E_yukawa, phi_yukawa = eigenvals(V_yukawa, ell, a, 150)
        if (E_yukawa[0] > 0):
            critical_coefficient = mu - 0.01
            break
    return critical_coefficient

if __name__ == "__main__":
    mode = input("Insert the name of the quantum system you want to examine (h-atom, fin-well or yukawa): ")
    match mode:
        case "h-atom":
            hEnergyPlot()
        case "fin-well": 
            criticalPotentials()
        case "yukawa":
            print(f"The determined critical coefficient is mu = {yukawaCriticals()}")
        case _:
            raise ValueError("string 'mode' is not h-atom, fin-well or yukawa")
