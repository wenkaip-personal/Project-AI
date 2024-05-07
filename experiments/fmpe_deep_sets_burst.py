import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import islice
from lampe.data import JointLoader
from lampe.utils import GDStep
from tqdm import trange

from fmpe_deep_sets import DeepSetFMPE, DeepSetFMPELoss
from generate_burst_data import simulate_burst

# Example usage
time = np.linspace(0, 1.0, 1000)  # Time array
max_ncomp = 5  # Maximum number of components

# Parameters for the burst model
t0_lower = time[0] + 0.1
t0_upper = time[-1] - 0.1
amp_lower = 5.0
amp_upper = 15.0
rise_val = 0.02
skew_val = 4

def generate_burst_params(ncomp):
    t0 = np.random.uniform(t0_lower, t0_upper, size=ncomp)
    amp = np.random.uniform(amp_lower, amp_upper, size=ncomp)
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
        ncomp = np.random.choice(range(1, max_ncomp+1))  # Randomly choose the number of components
        burstparams = generate_burst_params(ncomp)
        ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg, return_model=True, noise_type='gaussian')
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

ncomp_star = np.random.choice(range(1, max_ncomp+1))
burstparams_star = generate_burst_params(ncomp_star)
ymodel_star, ycounts_star = simulate_burst(time, ncomp_star, burstparams_star, ybkg, return_model=True, noise_type='gaussian')
theta_star = torch.from_numpy(burstparams_star).float()
x_star = torch.from_numpy(ycounts_star).float()

estimator.eval()
num_estimator.eval()

with torch.no_grad():
    log_p_theta_given_x_N = estimator.flow(x_star.unsqueeze(0)).log_prob(theta_star.unsqueeze(0))
    samples_theta_given_x_N = estimator.flow(x_star.unsqueeze(0)).sample((2**14,))
    # Squeeze out the singleton dimension
    samples_theta_given_x_N = samples_theta_given_x_N.squeeze(1)
    # Apply the condition without the extra dimension
    filtered_samples = samples_theta_given_x_N[torch.where(samples_theta_given_x_N[:, 0] <= t0_upper)]
    p_N_given_x = num_estimator(x_star.unsqueeze(0))

# Compute the joint posterior p(theta, N | x) = p(theta | x, N) * p(N | x)
joint_posterior = torch.exp(log_p_theta_given_x_N).squeeze() * p_N_given_x.squeeze()

print("Joint posterior p(theta, N | x):")
print(joint_posterior)

plt.rcParams.update({"font.size": 12})  # Set font size for plots

LABELS = [f'$t0_{i+1}$' for i in range(max_ncomp)] + \
         [f'$amp_{i+1}$' for i in range(max_ncomp)] + \
         [f'$rise_{i+1}$' for i in range(max_ncomp)] + \
         [f'$skew_{i+1}$' for i in range(max_ncomp)]

LOWER = torch.tensor([t0_lower]*max_ncomp + [amp_lower]*max_ncomp + [rise_val]*max_ncomp + [skew_val]*max_ncomp)
UPPER = torch.tensor([t0_upper]*max_ncomp + [amp_upper]*max_ncomp + [rise_val]*max_ncomp + [skew_val]*max_ncomp)

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

# Plot the joint posterior p(theta, N | x) as pair plots for various levels of N
for ncomp in range(1, max_ncomp+1):
    samples_theta_given_x_N_ncomp = samples_theta_given_x_N[torch.where(samples_theta_given_x_N[:, 0] <= t0_upper)].numpy()[:, :4*ncomp]
    fig, axes = plt.subplots(4, ncomp, figsize=(4*ncomp, 12))
    
    # Ensure axes are always 2-dimensional
    if ncomp == 1:
        axes = np.expand_dims(axes, axis=1)  # Expand the dimensions to make it 2D if ncomp is 1
    
    for i in range(4):
        for j in range(ncomp):
            idx = i*ncomp + j
            if i == 0:
                axes[i, j].hist(samples_theta_given_x_N_ncomp[:, idx], bins=30, density=True, alpha=0.7, color='orange')
                axes[i, j].axvline(theta_star[idx].item(), color='red', linestyle='--', linewidth=2)
                axes[i, j].set_xlabel(LABELS[idx])
                axes[i, j].set_ylabel('Density')
            else:
                axes[i, j].scatter(samples_theta_given_x_N_ncomp[:, j], samples_theta_given_x_N_ncomp[:, idx], s=5, alpha=0.5, color='blue')
                axes[i, j].axvline(theta_star[j].item(), color='red', linestyle='--', linewidth=1)
                axes[i, j].axhline(theta_star[idx].item(), color='red', linestyle='--', linewidth=1)
                axes[i, j].set_xlabel(LABELS[j])
                axes[i, j].set_ylabel(LABELS[idx])
    plt.suptitle(f'Joint Posterior p(theta, N={ncomp} | x)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'experiments/plots/joint_posterior_theta_N{ncomp}.png')  # Save the plot to the 'plots' folder
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
plt.savefig('experiments/plots/posterior_N.png')  # Save the plot to the 'plots' folder
plt.close()  # Close the current figure

# Plot the joint posterior p(theta, N | x) as a heatmap with brighter colors
t0_range = np.linspace(t0_lower, t0_upper, 100)
amp_range = np.linspace(amp_lower, amp_upper, 100)
N_range = np.arange(1, max_ncomp+1)

# Create a grid of t0 and amp values
t0_grid, amp_grid = np.meshgrid(t0_range, amp_range)

# Reshape the grid into a 2D array of shape (num_samples, 2)
theta_range = np.hstack([t0_grid.reshape(-1, 1), np.ones((len(t0_grid.ravel()), 3)) * [0, rise_val, skew_val]])

# Convert theta_range to a PyTorch tensor
theta_range = torch.from_numpy(theta_range).float()

# Pad theta_range to match the maximum theta dimension
pad_size = 4 * max_ncomp - theta_range.size(1)
if pad_size > 0:
    theta_range = F.pad(theta_range, (0, pad_size), "constant", 0)

# Compute log_p_theta_given_x_N for all theta values in the grid
log_p_theta_given_x_N = estimator.flow(x_star.unsqueeze(0)).log_prob(theta_range).detach().numpy()

# Reshape log_p_theta_given_x_N to match the grid shape
log_p_theta_given_x_N = log_p_theta_given_x_N.reshape(len(amp_range), len(t0_range))

# Compute the joint posterior for each value of N
joint_posterior_log = np.zeros((len(N_range), len(amp_range), len(t0_range)))

for i, N in enumerate(N_range):
    p_N_given_x_scalar_log = np.log(p_N_given_x[0, i].item())
    joint_posterior_log[i] = log_p_theta_given_x_N + p_N_given_x_scalar_log

# Normalize in log-space using log-sum-exp to prevent underflow
max_log_prob = np.max(joint_posterior_log, axis=(1, 2), keepdims=True)
exp_scaled = np.exp(joint_posterior_log - max_log_prob)
normalized_prob = exp_scaled / np.sum(exp_scaled, axis=(1, 2), keepdims=True)
log_normalized_prob = np.log(normalized_prob) + max_log_prob

# Average the log probabilities across N
mean_log_posterior = np.mean(log_normalized_prob, axis=0)

plt.figure(figsize=(8, 6))
plt.imshow(mean_log_posterior, origin='lower', aspect='auto', extent=[t0_lower, t0_upper, amp_lower, amp_upper], cmap='viridis')
plt.colorbar(label='Log Probability Density')
plt.xlabel('$t0$')
plt.ylabel('$amp$')
plt.title('Joint Log Posterior p(theta, N | x)')
plt.tight_layout()
plt.savefig('experiments/plots/joint_log_posterior_heatmap.png')  # Save the plot to the 'plots' folder
plt.close()  # Close the current figure