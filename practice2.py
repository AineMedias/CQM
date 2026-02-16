import numpy as np
import scipy
import matplotlib.pyplot as plt

h = 1
m = 0.5
w = 1
dx = 0.01

def harmonic_potential(x, E, m, w):
    return -E + (m*(w**2)*(x**2))/2

#fig, ax = plt.subplots()
# Numerov assumes E = 0
xs = np.linspace(-5.0, 5.0, 1000)
#ax.plot(xs, harmonic_potential(xs, 5, m, w))
#plt.show()

# Define Numerov propagator
def numerov_propagator(start, end, dx, E, m, w):
    # define variables
    x = np.arange(start, end, dx)
    psi = np.zeros(len(x))
    factor = -2*m/(h**2)
    p = factor * harmonic_potential(x, E, m, w)

    # Set initial line/boundary condition
    psi[0] = 0
    psi[1] = 1.0e-6

    # Execute Numerov method
    for idx in range(2, len(x)):
        psi[idx] = 2 * (1 - 5*(dx**2)* p[idx] /12)*psi[idx-1]
        psi[idx] -= (1 + (dx**2) * p[idx] / 12) *psi[idx-2]
        psi[idx] /= 1 + (dx**2) * p[idx] / 12

    return x, psi

# Run graph to test whether it works
start = -5
end = 5
E = 2.6
x, psi = numerov_propagator(start, end, dx, E, m, w)
#fig, ax = plt.subplots()
#ax.plot(x, psi)
#plt.show()

Egrid = np.linspace(0.0, 5.0, 120)
final_values = []
for Etest in Egrid:
    x, psi = numerov_propagator(start, end, dx, Etest, m, w)
    final_values.append(psi[-1])

# test node counting
def count_nodes():
    t = np.linspace(0, 2*np.pi, 1024)
    f = np.sin(t)
    for i in range(2, len(f)):
        if np.sign(f[i]) != np.sign(f[i-1]):
            print(t[i])

def test_e(x_start, x_end, dx, Etest, m, w):
    x, psi = numerov_propagator(x_start, x_end, dx, Etest, m, w)
    nodes = 0
    for i in range(2, len(psi)):
        if np.sign(psi[i]) != np.sign(psi[i-1]):
            nodes += 1
    return nodes

plt.plot(Egrid, [test_e(-5, 5, dx, E, m, w) for E in Egrid])
plt.show()

Emin = 0
Emax = 5
target_nodes = 2
# insert Numerov logic
for j in range(10):
    Etest = 0.5* (Emin + Emax)
    nodes = 0
    if test_e(-5, 5, dx, Etest, m, w) >= target_nodes + 1:
        Emax = Etest
    else:
        Emin = Etest
    print(f"j {j:2d}, nodes: {nodes:2d}, Etest: {Etest:7.5g}, Emin, Emax: {Emin:7.5g}, {Emax:7.5g}")

# Finally, design a function to iterate over a defined phase-space (E-space) and find stable solutions in a range.
def numerov_iterator(start, end, precision, xstart = -5, xend = 5):
    ranges = []
    errors = []
    for energy in np.arange(start, end, precision):
        position, wave = numerov_propagator(xstart, xend, dx, energy, m, w)










