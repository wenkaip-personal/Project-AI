import torch
import numpy as np
import matplotlib.pyplot as plt

from generate_burst_data import simulate_burst

# Parameters for the burst model
t0_lower = 0.1
t0_upper = 0.9
amp_lower = 2
amp_upper = 5
rise_val = 0.02
skew_val = 4

def generate_burst_params(ncomp, amp_lower, amp_upper, rise_val, skew_val):
    t0 = np.random.uniform(t0_lower, t0_upper, size=ncomp)
    amp = np.random.uniform(amp_lower, amp_upper, size=ncomp)
    rise = np.ones(ncomp) * rise_val
    skew = np.ones(ncomp) * skew_val
    return np.hstack([t0, amp*10, rise, skew])

def visualize_burst(time, ycounts, ymodel, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, ycounts, color="black", label="Poisson model counts")
    ax.plot(time, ymodel, color="orange", label="Noise-free model")
    ax.set_xlabel("Time [arbitrary units]")
    ax.set_ylabel("Flux [arbitrary units]")
    ax.set_xlim(time[0], time[-1])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Example 1: Varying the number of components
time = np.linspace(0, 1.0, 1000)  # Time array
ybkg = 1.0  # Background flux

for ncomp in [1, 3, 5]:
    burstparams = generate_burst_params(ncomp, amp_lower, amp_upper, rise_val, skew_val)
    ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg * 10, return_model=True)
    theta = torch.from_numpy(burstparams).float()
    x = torch.from_numpy(ycounts).float()
    visualize_burst(time, ycounts, ymodel, f"Number of Components: {ncomp}")

# Example 2: Varying the amplitude range
ncomp = 3
ybkg = 1.0

for amp_lower, amp_upper in [(1, 3), (5, 10), (10, 20)]:
    burstparams = generate_burst_params(ncomp, amp_lower, amp_upper, rise_val, skew_val)
    ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg * 10, return_model=True)
    theta = torch.from_numpy(burstparams).float()
    x = torch.from_numpy(ycounts).float()
    visualize_burst(time, ycounts, ymodel, f"Amplitude Range: [{amp_lower}, {amp_upper}]")

# Example 3: Varying the rise time
ncomp = 3
ybkg = 1.0

for rise_val in [0.01, 0.05, 0.1]:
    burstparams = generate_burst_params(ncomp, amp_lower, amp_upper, rise_val, skew_val)
    ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg * 10, return_model=True)
    theta = torch.from_numpy(burstparams).float()
    x = torch.from_numpy(ycounts).float()
    visualize_burst(time, ycounts, ymodel, f"Rise Time: {rise_val}")

# Example 4: Varying the skewness
ncomp = 3
ybkg = 1.0

for skew_val in [1, 5, 10]:
    burstparams = generate_burst_params(ncomp, amp_lower, amp_upper, rise_val, skew_val)
    ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg * 10, return_model=True)
    theta = torch.from_numpy(burstparams).float()
    x = torch.from_numpy(ycounts).float()
    visualize_burst(time, ycounts, ymodel, f"Skewness: {skew_val}")

# Example 5: Varying the background level
ncomp = 3

for ybkg in [0.5, 1.0, 2.0]:
    burstparams = generate_burst_params(ncomp, amp_lower, amp_upper, rise_val, skew_val)
    ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg * 10, return_model=True)
    theta = torch.from_numpy(burstparams).float()
    x = torch.from_numpy(ycounts).float()
    visualize_burst(time, ycounts, ymodel, f"Background Level: {ybkg}")

# Example 6: Varying the time range
ncomp = 3
ybkg = 1.0

for tmin, tmax in [(0, 0.5), (0.5, 1.0), (0, 2.0)]:
    time = np.linspace(tmin, tmax, 1000)
    t0_lower = time[0] + 0.1
    t0_upper = time[-1] - 0.1
    burstparams = generate_burst_params(ncomp, amp_lower, amp_upper, rise_val, skew_val)
    ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg * 10, return_model=True)
    theta = torch.from_numpy(burstparams).float()
    x = torch.from_numpy(ycounts).float()
    visualize_burst(time, ycounts, ymodel, f"Time Range: [{tmin}, {tmax}]")