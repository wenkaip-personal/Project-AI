import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from itertools import islice
from lampe.data import JointLoader
from lampe.plots import corner, mark_point, nice_rc
from lampe.utils import GDStep
from tqdm import trange

from fmpe_deep_sets import DeepSetFMPE, DeepSetFMPELoss
from generate_burst_data import simulate_burst

# Example usage
time = np.linspace(0, 1.0, 1000)  # Time array
max_ncomp = 2  # Maximum number of components

# Parameters for the burst model
t0_lower = time[0] + 0.1
t0_upper = time[-1] - 0.1
amp_val = 3
rise_val = 0.02
skew_val = 4

def generate_burst_params(ncomp):
    t0 = np.random.uniform(t0_lower, t0_upper, size=ncomp)
    amp = np.ones(ncomp) * amp_val
    rise = np.ones(ncomp) * rise_val
    skew = np.ones(ncomp) * skew_val
    return np.hstack([t0, amp, rise, skew])

ybkg = 1.0  # Background flux

class CategoricalModel(nn.Module):
    def __init__(self, x_dim, max_components):
        super(CategoricalModel, self).__init__()
        self.max_components = max_components
        self.fc1 = nn.Linear(x_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, max_components)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.softmax(x)
        return probs

def categorical_loss(probs, targets):
    targets = targets.long()  # Ensure targets are long type for indexing
    return -torch.log(probs[torch.arange(probs.size(0)), targets]).mean()

estimator = DeepSetFMPE(theta_dim=4 * max_ncomp, x_dim=1000, freqs=5)
loss = DeepSetFMPELoss(estimator)
optimizer = optim.Adam(estimator.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 128)
step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

num_estimator = CategoricalModel(x_dim=1000, max_components=max_ncomp)
num_optimizer = optim.Adam(num_estimator.parameters(), lr=1e-3)
num_scheduler = optim.lr_scheduler.CosineAnnealingLR(num_optimizer, 128)
num_step = GDStep(num_optimizer, clip=1.0)

estimator.train()
num_estimator.train()

for epoch in (bar := trange(128, unit='epoch')):
    losses = []
    num_losses = []

    for _ in range(256):  # 256 batches per epoch
        ncomp = np.random.choice([1, 2])  # Randomly choose the number of components
        burstparams = generate_burst_params(ncomp)
        ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg * 10, return_model=True)
        theta = torch.from_numpy(burstparams).float()
        x = torch.from_numpy(ycounts).float()
        num_target = torch.tensor([ncomp - 1], dtype=torch.long)  # Adjust the target to 0-based index, ensure long type


        # Ensure theta has the maximum theta dimension by padding
        pad_size = 4 * max_ncomp - theta.size(0)
        if pad_size > 0:
            theta = F.pad(theta, (0, pad_size), "constant", 0)

        losses.append(step(loss(theta, x)))
        num_losses.append(num_step(categorical_loss(num_estimator(x.unsqueeze(0)), num_target)))  # Ensure x is batched

    bar.set_postfix(loss=torch.stack(losses).mean().item(), num_loss=torch.stack(num_losses).mean().item())

ncomp_star = np.random.choice([1, 2])
burstparams_star = generate_burst_params(ncomp_star)
ymodel_star, ycounts_star = simulate_burst(time, ncomp_star, burstparams_star, ybkg * 10, return_model=True)
theta_star = torch.from_numpy(burstparams_star).float()
x_star = torch.from_numpy(ycounts_star).float()

estimator.eval()
num_estimator.eval()

with torch.no_grad():
    log_p_t0_given_x_N = estimator.flow(x_star.unsqueeze(0)).log_prob(theta_star.unsqueeze(0))
    samples_t0_given_x_N = estimator.flow(x_star.unsqueeze(0)).sample((2**14,))
    p_N_given_x = num_estimator(x_star.unsqueeze(0))

# Compute the joint posterior p(t0, N | x) = p(t0 | x, N) * p(N | x)
joint_posterior = torch.exp(log_p_t0_given_x_N).squeeze() * p_N_given_x.squeeze()

print("Joint posterior p(t0, N | x):")
print(joint_posterior)

plt.rcParams.update(nice_rc(latex=True))  # nicer plot settings

LABELS = [f'$t0_{i+1}$' for i in range(max_ncomp)] + \
         [f'$amp_{i+1}$' for i in range(max_ncomp)] + \
         [f'$rise_{i+1}$' for i in range(max_ncomp)] + \
         [f'$skew_{i+1}$' for i in range(max_ncomp)]

LOWER = torch.tensor([t0_lower]*max_ncomp + [amp_val]*max_ncomp + [rise_val]*max_ncomp + [skew_val]*max_ncomp)
UPPER = torch.tensor([t0_upper]*max_ncomp + [amp_val]*max_ncomp + [rise_val]*max_ncomp + [skew_val]*max_ncomp)

# Plot the synthetic data with brighter colors
plt.figure(figsize=(8, 4))
plt.plot(time, ymodel_star, label='True Model', color='red', linewidth=2)
plt.plot(time, ycounts_star, label='Observed Counts', color='blue', linewidth=2, alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title(f'Synthetic Data (N={ncomp_star})')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('experiments/plots/synthetic_data.png')  # Save the plot to the 'plots' folder
plt.close()  # Close the current figure

# Plot the joint posterior p(t0, N | x) with brighter colors
fig, axes = plt.subplots(1, max_ncomp, figsize=(12, 4))
for ncomp, ax in enumerate(axes, start=1):
    ax.hist(samples_t0_given_x_N[ncomp-1][:, 0].numpy(), bins=50, density=True, alpha=0.7, label=f'$p(t0 | x, N={ncomp})$', color='orange')
    ax.axvline(theta_star[0].item(), color='red', linestyle='--', label='True $t0$', linewidth=2)
    ax.set_xlabel(f'$t0$ (N={ncomp})')
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior p(t0 | x, N={ncomp})')
    ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('experiments/plots/joint_posterior_t0_N.png')  # Save the plot to the 'plots' folder
plt.close()  # Close the current figure

# Plot the posterior p(N | x) with brighter colors
plt.figure(figsize=(6, 4))
plt.bar(range(1, max_ncomp+1), p_N_given_x.squeeze().numpy(), alpha=0.7, color='green')
plt.axvline(ncomp_star, color='red', linestyle='--', label='True N', linewidth=2)
plt.xlabel('Number of Components (N)')
plt.ylabel('Probability')
plt.title('Posterior p(N | x)')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('experiments/plots/joint_posterior_N.png')  # Save the plot to the 'plots' folder
plt.close()  # Close the current figure

# Plot the joint posterior p(t0, N | x) as a heatmap with brighter colors
t0_range = np.linspace(t0_lower, t0_upper, 100)
N_range = np.arange(1, max_ncomp+1)
joint_posterior = np.zeros((len(N_range), len(t0_range)))

for i, N in enumerate(N_range):
    # Ensure theta has the maximum theta dimension by padding
    theta_range = torch.from_numpy(np.hstack([t0_range.reshape(-1, 1), np.ones((len(t0_range), 3)) * [amp_val, rise_val, skew_val]])).float()
    pad_size = 4 * max_ncomp - theta_range.size(1)
    if pad_size > 0:
        theta_range = F.pad(theta_range, (0, pad_size), "constant", 0)

    log_p_t0_given_x_N = estimator.flow(x_star.unsqueeze(0)).log_prob(theta_range).detach().numpy()
    p_N_given_x_scalar = p_N_given_x[0, i].item()  # Extract the scalar value from p_N_given_x[i]
    joint_posterior[i] = np.exp(log_p_t0_given_x_N) * p_N_given_x_scalar

plt.figure(figsize=(8, 6))
plt.imshow(joint_posterior, origin='lower', aspect='auto', extent=[t0_lower, t0_upper, 0.5, max_ncomp+0.5], cmap='viridis')
plt.colorbar(label='Probability Density')
plt.xlabel('$t0$')
plt.ylabel('Number of Components (N)')
plt.title('Joint Posterior p(t0, N | x)')
plt.tight_layout()
plt.savefig('experiments/plots/joint_posterior_heatmap.png')  # Save the plot to the 'plots' folder
plt.close()  # Close the current figure
