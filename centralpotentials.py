# @2026 Aine Productions.
# Aine Productions is not responsible for shenanigans resulting from the wrongful acquisition, 
# appropriation or otherwise unintended use of this program.
# Aine Productions, however, IS responsible for shenanigans resulting from the *rightful* use of this program.

# Program for finding energy eigenstates and either plotting them as a function of n,
# or using them to (attempt to) figure out critical potential values where relevant.


import numpy as np
from tqdm import tqdm
from scipy import integrate, linalg
from matplotlib import pyplot as plt

# Define all SI-constants here.
# For ease of usability, this program uses natural units.
hbar = 1
m0 = 1
e = 1
eps_0 = 1 / (4*np.pi)
r0 = (4 * np.pi * eps_0 * (hbar**2)) / (m0*(e**2)) # Bohr radius

# Define variables here
ell = 0
a = 50 * r0
n_max = 200


# Define centrifugal potential
def E_cen(ell, r): 
    return -1 * ( ell*(ell+1) * hbar**2 ) / (2*m0*r**2)

# Determine eigenvalues from a defined potential (omega(n^3))
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

# First step: Plot energies of hydrogen atom to predicted energy value
def hEnergyPlot():
    # Determine energies of Hydrogen atom potential
    # V_h = lambda r: -e**2 / (4*np.pi*eps_0*r)
    V_h = lambda r: -1 / r
    E_hatom, phi_hatom = eigenvals(V_h, ell, a, n_max)

    n_hatom = range(0, n_max)
    # Compare to Rydberg energies defined in eV ( E = -13.6/n^2 ), but expressed in atomic units (unit of energy is roughly 27.211 eV) 
    # by plotting both it and the acquired energies
    E_rydberg = [(-13.6 / n**2) / 27.211 for n in range(1, n_max)]


    plt.plot(n_hatom, E_hatom, "-")
    plt.plot(range(1, n_max), E_rydberg, "o")
    plt.xlim(0, 20)
    plt.ylim(-1.25, 1.25)
    plt.ylabel("energy [E_h]")
    plt.xlabel("n")
    plt.title(f"Energy eigenvalues of hydrogen atom at a = {a/r0}a0")
    plt.legend(["Program energies", "numeric results"])
    plt.show()

# solve finite spherical well
def finiteSphericalWell(singlestate=0, v0=0, b=0.15*a):
    # Second step (default): determine critical potentials
    if (singlestate == 0):
        critical_potentials = np.zeros(round(a))
        # search in ranges of b and V0
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
    # if singlestate is on, the function will simply plot the eigenvalues of the finite spherical well as a function of n
    elif (singlestate == 1):
        def V_sphere(r):
            if (r > b or r < 0):
                return 0
            else:
                return (-1 * v0)
                    
        E_sphere, phi_sphere = eigenvals(V_sphere, ell, a, n_max)

        n_sphere = range(0, n_max)
        plt.plot(n_sphere, E_sphere, "o-")
        plt.xlim(0, 25)
        plt.ylim(-1.25, 1.25)
        plt.ylabel("energy E [27.211 eV]")
        plt.xlabel("n")
        plt.title(f"Energy eigenvalues of finite spherical well with V0 = {v0}, b = {b}")
        plt.show()
        return E_sphere

# Yukawa potential
def yukawaPotential(mode = 0, u = 0):
    A = 1 # interaction strength
    # Step three (default): determine critical coefficient of Yukawa potential
    criticalCoefficientFound = False
    if (mode != 1):
        critical_coefficient = 0
        mu = 0.90
        while (mu in np.arange(0.90, 1.25, 0.01) and criticalCoefficientFound == False):
            print(f"Now examining mu = {mu}")
            V_yukawa = lambda r: -1 * A * (np.e ** (-1 * mu * r / r0)) / r
            E_yukawa, phi_yukawa = eigenvals(V_yukawa, ell, a, 150)
            if (E_yukawa[0] > 0):
                critical_coefficient = mu - 0.01
                print("Critical coefficient found")
                criticalCoefficientFound = True
            mu += 0.01
        # return 0 if code execution fails to find a critical coefficient
        return critical_coefficient
    # if singlestate is on (= 1), the function will simply plot the eigenvalues of the Yukawa potential as a function of n
    else:
        V_yukawa = lambda r: -1 * A * (np.e ** (-1 * u * r / r0)) / r
        E_yukawa, phi_yukawa = eigenvals(V_yukawa, ell, a, n_max)

        n_yukawa = range(0, n_max)
        plt.plot(n_yukawa, E_yukawa, "o-")
        plt.xlim(0, 25)
        plt.ylim(-1.25, 1.25)
        plt.ylabel("energy E [27.211 eV]")
        plt.xlabel("n")
        plt.title(f"Energy eigenvalues of Yukawa potential at mu = {u}")
        plt.show()
        print(f"The lowest-energy eigenvalue of this quantum system is {E_yukawa[0]} Hartree units")


if __name__ == "__main__":
    sys = input("Insert the name of the quantum system you want to examine (h-atom, fin-well or yukawa): ")
    mode = int(input("Do you want to examine a system at a particular potential value [press 1], or go with default settings? [press 0]: "))
    print(f"Attempting to start mode '{sys}'")
    if (mode != 0 and mode != 1):
        raise ValueError(f"value 'mode' must be 0 or 1, not {mode}")
    else:
        match sys:
            case "h-atom":
                print("Starting mode 'h-atom', automatically set to default mode")
                hEnergyPlot()
            case "fin-well":
                print("Starting mode 'fin-well'")
                if (mode == 0):
                    finiteSphericalWell()
                else:
                    v0 = float(input("Please insert the potential depth: "))
                    b = float(input(f"Please insert the well width (b) in Bohr radii (size a = {a}): "))
                    finiteSphericalWell(mode, v0, b)
            case "yukawa":
                print("Starting mode 'yukawa'")
                if (mode == 0):
                    print(f"The determined critical coefficient is mu = {yukawaPotential(mode)}")
                else:
                    u = float(input("Please insert the constant mu for the relevant Yukawa potential: "))
                    yukawaPotential(mode, u)
            case _:
                raise ValueError(f"string '{sys}' is not h-atom, fin-well or yukawa")
    print("Code execution complete!")
