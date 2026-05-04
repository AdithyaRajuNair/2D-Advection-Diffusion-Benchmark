import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def extract_step(filename):
    match = re.search(r"output_step_(\d+)\.csv", filename)
    return int(match.group(1)) if match else -1

def load_velocity(filename, N):
    data = np.loadtxt(filename, delimiter=",")
    
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            u[i, j] = data[i, 2*j]
            v[i, j] = data[i, 2*j + 1]
    
    return u, v


# Sort temperature files
files = sorted(glob.glob("output_step_*.csv"), key=extract_step)

N = 512
stride = 8  # downsampling for quiver

for f in files:
    step = extract_step(f)

    # Load temperature
    temp = np.loadtxt(f, delimiter=",")

    # Load velocity
    vel_file = f"velocity_step_{step}.csv"
    u, v = load_velocity(vel_file, N)

    # Grid
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    # Plot heatmap
    plt.imshow(temp, cmap="hot", origin="lower")

    # Overlay quiver
    plt.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        color="cyan", scale=50
    )

    plt.title(f"Step {step}")
    plt.colorbar()
    plt.pause(0.1)
    plt.clf()

plt.show()