import numpy as np
from tqdm import tqdm
import scipy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display

N = 256 # grid size
L = 20 # length
m = 0.5 # mass
h = 1

x_grid = np.linspace(-L/2, L/2, N+1); x_grid = x_grid[0:-1]
y_grid = np.linspace(-L/2, L/2, N+1); y_grid = y_grid[0:-1]
dx = x_grid[1] - x_grid[0]

# produce N by N grid
X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

# execute a FFT on said grid, returning frequencies (x, y => p_x, p_y)
kx_grid = np.fft.fftfreq(N, d=dx) * (2 * np.pi)
ky_grid = np.fft.fftfreq(N, d=dx) * (2 * np.pi)

KX, KY = np.meshgrid(kx_grid, ky_grid, indexing='xy')

# define kinetic energy, p^2/2m
T_k = h**2 * (KX**2 + KY**2) / (2*m)

# Next, define atomic potential energy
V_atom = -1 / (X**2 + Y**2)


# Next, define a propagation step
def propagation_step(psi, T, V, dt):
    # Initiate first kick
    psi *= np.exp(-1.0) * V * dt / (h*2)
    # Then FFT to P-space
    psi = np.fft.fft2(psi)
    # Apply drift operator
    psi *= np.exp(-1.0) * T * dt / h
    # then iFFT back to X-space and apply second kick
    psi = np.fft.ifft2(psi)
    psi *= np.exp(-1.0) * V * dt / (h*2)

    return psi

# calculate ground state using T-dep splitfunction to bypass numerical disadvantages

sigma = 2
psi_gs = np.exp(-((X**2) + (Y**2)) / (2*sigma**2)) + 0.0j
psi_gs /= np.sqrt(np.sum(np.abs(psi_gs)**2) * dx**2)

dtau = -0.01j

for i in range(1000):
    oldPsi = psi_gs.copy()
    psi_gs = propagation_step(psi_gs, T_k, V_atom, dtau)
    psi_gs /= np.sqrt(np.sum(np.abs(psi_gs)**2) * dx**2)

    diff = np.sum(np.abs(psi_gs - oldPsi) ** 2)
    if diff < 1.0e-6:
        break

print(i)
plt.imshow(np.abs(psi_gs)**2)
plt.show()

# print(x_grid.shape)
# V_1 = V_atom + X
V_1 = lambda t : V_atom + 0.25 * np.cos(0.5 * t) * X

dt = 0.01
psi = psi_gs.copy()
fig, ax = plt.subplots()
im = ax.imshow(np.abs(psi)**2)

def update(frame):
    global psi
    global t

    for j in range(25):
        psi = propagation_step(psi, T_k, V_1, dt)
    t += dt * (j + 1)

    im.set_array(ax.imshow(np.abs(psi)**2))

    return [im]

anim = FuncAnimation(fig, update, frames = 10, blit=True)
anim.save("animation.gif", writer=PillowWriter(fps=15))
plt.close()
    
display(Image(filename = "animation.gif"))