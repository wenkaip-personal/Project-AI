import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import corner
import numpy as np
from itertools import islice
from lampe.data import JointLoader
from lampe.plots import nice_rc
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
amp_val = 5.0
rise_val = 0.02
skew_val = 4

def generate_burst_params(ncomp):
    t0 = np.random.uniform(t0_lower, t0_upper, size=ncomp)
    amp = np.ones(ncomp) * amp_val
    rise = np.ones(ncomp) * rise_val
    skew = np.ones(ncomp) * skew_val
    return np.hstack([t0, amp*10, rise, skew])

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
        # ncomp = np.random.choice(range(1, max_ncomp+1))  # Randomly choose the number of components
        ncomp = 2
        burstparams = generate_burst_params(ncomp)
        ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg*10, return_model=True, noise_type='gaussian')
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

# ncomp_star = np.random.choice(range(1, max_ncomp+1))
ncomp_star = 2
burstparams_star = generate_burst_params(ncomp_star)
ymodel_star, ycounts_star = simulate_burst(time, ncomp_star, burstparams_star, ybkg*10, return_model=True, noise_type='gaussian')
theta_star = torch.from_numpy(burstparams_star).float()
x_star = torch.from_numpy(ycounts_star).float()

# Ensure theta_star has the maximum theta dimension by padding
pad_size = 4 * max_ncomp - theta_star.size(0)
if pad_size > 0:
    theta_star = F.pad(theta_star, (0, pad_size), "constant", 0)

estimator.eval()
num_estimator.eval()

with torch.no_grad():
    log_p_theta_given_x_N = estimator.flow(x_star).log_prob(theta_star)
    samples_theta_given_x_N = estimator.flow(x_star).sample((2**14,))

    p_N_given_x = num_estimator(x_star)

# Compute the joint posterior p(theta, N | x) = p(theta | x, N) * p(N | x)
joint_posterior = torch.exp(log_p_theta_given_x_N).squeeze() * p_N_given_x.squeeze()

print("Joint posterior p(theta, N | x):")
print(joint_posterior)

plt.rcParams.update(nice_rc(latex=True))  # nicer plot settings

# Plot the synthetic data
fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(time, ycounts_star, color="black", label="Gaussian model counts")
ax.plot(time, ymodel_star, color="orange", label="Noise-free model")
ax.set_xlabel("Time [arbitrary units]")
ax.set_ylabel("Flux [arbitrary units]")
ax.set_title(f'Synthetic Data (N={ncomp_star})')
ax.set_xlim(time[0], time[-1])
ax.legend()
fig.tight_layout() 
fig.savefig('experiments/plots/synthetic_data.png')  # Save the plot to the 'plots' folder
plt.close(fig)  # Close the current figure

# Plot the posterior p(N | x)
fig, ax = plt.subplots()
ax.plot(range(1, max_ncomp+1), p_N_given_x.squeeze().numpy(), marker='o')
ax.set_xlabel('Number of Components N')
ax.set_ylabel('Posterior Probability p(N | x)')
ax.set_title('Posterior Distribution of N')
fig.tight_layout()
fig.savefig('experiments/plots/posterior_N.png')
plt.close(fig)

# Plot the observation x
fig, ax = plt.subplots()
ax.plot(time, x_star.numpy(), color="blue", label="Observed counts")
ax.set_xlabel("Time [arbitrary units]")
ax.set_ylabel("Counts")
ax.set_title("Observation x")
ax.legend()
fig.tight_layout()
fig.savefig('experiments/plots/observation_x.png')
plt.close(fig)

# Plot the joint posterior p(theta, N | x) as pair plots for various levels of N
for n in range(1, max_ncomp + 1):
    samples_n = samples_theta_given_x_N[:, :4*n].numpy()  # Convert tensor to numpy array
    labels_n = [f'$t0_{i+1}$' for i in range(n)]
    
    # Filter out negative values for t0
    mask = (samples_n[:, :n] > 0).all(axis=1)
    samples_n = samples_n[mask]
    
    fig = corner.corner(samples_n[:, :n], labels=labels_n, truths=burstparams_star[:n], show_titles=True, 
                        plot_contours=True, levels=(0.68, 0.95), color='blue')
    fig.suptitle(f'Joint Posterior for N={n}', fontsize=16, y=1.05)  # Adjust the title position
    fig.savefig(f'experiments/plots/joint_posterior_N{n}.png')
    plt.close(fig)

# Sampling from p(N | x) and p(theta | N, x)
num_samples = 10000
Ns = torch.multinomial(p_N_given_x, num_samples, replacement=True).squeeze() + 1  # Sample N from p(N | x)

# Create a mask for the theta tensor based on sampled N values
theta_mask = torch.zeros((num_samples, max_ncomp * 4), dtype=torch.bool)
for i, n in enumerate(Ns):
    theta_mask[i, :n*4] = True

# Sample theta from p(theta | N, x) using Flow Matching
thetas = torch.zeros((num_samples, max_ncomp * 4))
for n in range(1, max_ncomp + 1):
    mask_n = Ns == n
    if mask_n.any():
        x_n = x_star.repeat(mask_n.sum(), 1)
        thetas_n = estimator.flow(x_n).sample((1,))
        thetas[mask_n] = thetas_n.squeeze()

# Apply the mask to the sampled thetas
thetas = thetas[theta_mask].view(num_samples, -1)

# Group the sampled thetas by N
grouped_thetas = {n: thetas[Ns == n] for n in range(1, max_ncomp + 1)}

# Plot the joint posterior p(theta, N | x) using the sampled thetas and Ns
for n in range(1, max_ncomp + 1):
    samples_n = grouped_thetas[n].numpy()
    labels_n = [f'$t0_{i+1}$' for i in range(n)]
    
    # t0 must be positive
    mask = (samples_n[:, :n] > 0).all(axis=1)
    samples_n = samples_n[mask]
    
    if samples_n.shape[0] > 0:  # Check if there are any samples after filtering
        fig = corner.corner(samples_n[:, :n], labels=labels_n, truths=burstparams_star[:n], show_titles=True, 
                            plot_contours=True, levels=(0.68, 0.95), color='blue')
        fig.suptitle(f'Joint Posterior for N={n} (Sampled)', fontsize=16, y=1.05)  # Adjust the title position
        fig.savefig(f'experiments/plots/joint_posterior_sampled_N{n}.png')
        plt.close(fig)
    else:
        print(f"No valid samples for N={n} after filtering.")

