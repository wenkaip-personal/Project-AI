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
    t0 = np.random.uniform(t0_lower + 0.15, t0_upper - 0.15, size=ncomp)
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

train_losses = []
train_num_losses = []
val_losses = []
val_num_losses = []

for epoch in (bar := trange(128, unit='epoch')):
    epoch_train_losses = []
    epoch_train_num_losses = []
    epoch_val_losses = []
    epoch_val_num_losses = []

    for _ in range(256):  # 256 batches per epoch
        # Training step
        ncomp = np.random.choice(range(1, max_ncomp+1))  # Randomly choose the number of components
        # ncomp = 2
        burstparams = generate_burst_params(ncomp)
        ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg*10, return_model=True, noise_type='gaussian')
        theta = torch.from_numpy(burstparams).float()
        x = torch.from_numpy(ycounts).float()
        num_target = torch.tensor([ncomp - 1], dtype=torch.long)  # Adjust the target to 0-based index, ensure long type

        # Ensure theta has the maximum theta dimension by padding
        pad_size = 4 * max_ncomp - theta.size(0)
        if pad_size > 0:
            theta = F.pad(theta, (0, pad_size), "constant", 0)

        epoch_train_losses.append(step(loss(theta, x)))
        epoch_train_num_losses.append(num_step(categorical_loss(num_estimator(x.unsqueeze(0)), num_target)))  # Ensure x is batched

        # Validation step
        with torch.no_grad():
            val_ncomp = np.random.choice(range(1, max_ncomp+1))  # Randomly choose the number of components
            # val_ncomp = 2
            val_burstparams = generate_burst_params(val_ncomp)
            val_ymodel, val_ycounts = simulate_burst(time, val_ncomp, val_burstparams, ybkg*10, return_model=True, noise_type='gaussian')
            val_theta = torch.from_numpy(val_burstparams).float()
            val_x = torch.from_numpy(val_ycounts).float()
            val_num_target = torch.tensor([val_ncomp - 1], dtype=torch.long)  # Adjust the target to 0-based index, ensure long type

            # Ensure val_theta has the maximum theta dimension by padding
            val_pad_size = 4 * max_ncomp - val_theta.size(0)
            if val_pad_size > 0:
                val_theta = F.pad(val_theta, (0, val_pad_size), "constant", 0)

            epoch_val_losses.append(loss(val_theta, val_x))
            epoch_val_num_losses.append(categorical_loss(num_estimator(val_x.unsqueeze(0)), val_num_target))  # Ensure val_x is batched

    train_losses.append(torch.stack(epoch_train_losses).mean().item())
    train_num_losses.append(torch.stack(epoch_train_num_losses).mean().item())
    val_losses.append(torch.stack(epoch_val_losses).mean().item())
    val_num_losses.append(torch.stack(epoch_val_num_losses).mean().item())

    bar.set_postfix(train_loss=train_losses[-1], train_num_loss=train_num_losses[-1], val_loss=val_losses[-1], val_num_loss=val_num_losses[-1])

# Plot the loss curves
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.savefig('experiments/plots_test/loss_curves_random_ncomp.png')
plt.close()

plt.figure()
plt.plot(train_num_losses, label='Train Categorical Loss')
plt.plot(val_num_losses, label='Validation Categorical Loss')
plt.xlabel('Epoch')
plt.ylabel('Categorical Loss')
plt.title('Training and Validation Categorical Loss Curves')
plt.legend()
plt.savefig('experiments/plots_test/num_loss_curves_random_ncomp.png')
plt.close()

ncomp_star = np.random.choice(range(1, max_ncomp+1))
# ncomp_star = 2
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

# Check if inputs are approximately normally distributed
def check_normality(tensor, name):
    mean = tensor.mean().item()
    std = tensor.std().item()
    print(f"{name} - Mean: {mean}, Std: {std}")

check_normality(x, "x")
check_normality(val_x, "val_x")
check_normality(x_star, "x_star")

# Check if outputs are exploding or going to zero
def check_distribution(tensor, name):
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    print(f"{name} - Min: {min_val}, Max: {max_val}")

check_distribution(torch.exp(log_p_theta_given_x_N), "p_theta_given_x_N")
check_distribution(p_N_given_x, "p_N_given_x")

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
fig.savefig('experiments/plots_test/synthetic_data.png')
plt.close(fig)  # Close the current figure

# Plot the posterior p(N | x)
fig, ax = plt.subplots()
ax.plot(range(1, max_ncomp+1), p_N_given_x.squeeze().numpy(), marker='o')
ax.set_xlabel('Number of Components N')
ax.set_ylabel('Posterior Probability p(N | x)')
ax.set_title('Posterior Distribution of N')
fig.tight_layout()
fig.savefig('experiments/plots_test/posterior_N.png')
plt.close(fig)

# Plot the observation x
fig, ax = plt.subplots()
ax.plot(time, x_star.numpy(), color="blue", label="Observed counts")
ax.set_xlabel("Time [arbitrary units]")
ax.set_ylabel("Counts")
ax.set_title("Observation x")
ax.legend()
fig.tight_layout()
fig.savefig('experiments/plots_test/observation_x.png')
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
    fig.savefig(f'experiments/plots_test/joint_posterior_N{n}.png')
    plt.close(fig)

# # Sampling from p(N | x) and p(theta | N, x)
# num_samples = 10000
# Ns = torch.multinomial(p_N_given_x, num_samples, replacement=True).squeeze() + 1  # Sample N from p(N | x)

# # Create a mask for the theta tensor based on sampled N values
# theta_mask = torch.zeros((num_samples, max_ncomp * 4), dtype=torch.bool)
# for i, n in enumerate(Ns):
#     theta_mask[i, :n*4] = True

# # Sample theta from p(theta | N, x) using Flow Matching
# thetas = torch.zeros((num_samples, max_ncomp * 4))
# for n in range(1, max_ncomp + 1):
#     mask_n = Ns == n
#     if mask_n.any():
#         x_n = x_star.repeat(mask_n.sum(), 1)
#         thetas_n = estimator.flow(x_n).sample((1,))
#         thetas[mask_n] = thetas_n.squeeze()

# # Apply the mask to the sampled thetas
# thetas = thetas[theta_mask].view(num_samples, -1)

# # Group the sampled thetas by N
# grouped_thetas = {n: thetas[Ns == n] for n in range(1, max_ncomp + 1)}

# # Plot the joint posterior p(theta, N | x) using the sampled thetas and Ns
# for n in range(1, max_ncomp + 1):
#     samples_n = grouped_thetas[n].numpy()
#     labels_n = [f'$t0_{i+1}$' for i in range(n)]
    
#     # t0 must be positive
#     mask = (samples_n[:, :n] > 0).all(axis=1)
#     samples_n = samples_n[mask]
    
#     if samples_n.shape[0] > 0:  # Check if there are any samples after filtering
#         fig = corner.corner(samples_n[:, :n], labels=labels_n, truths=burstparams_star[:n], show_titles=True, 
#                             plot_contours=True, levels=(0.68, 0.95), color='blue')
#         fig.suptitle(f'Joint Posterior for N={n} (Sampled)', fontsize=16, y=1.05)  # Adjust the title position
#         fig.savefig(f'experiments/plots_test/joint_posterior_sampled_N{n}.png')
#         plt.close(fig)
#     else:
#         print(f"No valid samples for N={n} after filtering.")

# Draw several samples from the posterior and make peak plots
num_peak_samples = 5
peak_samples = estimator.flow(x_star).sample((num_peak_samples,))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i in range(num_peak_samples):
    sample_params = peak_samples[i].numpy()
    print(f"Sample {i+1} parameters: {sample_params}")  # Debug print
    sample_ymodel, _ = simulate_burst(time, ncomp_star, sample_params, ybkg*10, return_model=True, noise_type='gaussian')
    ax.plot(time, sample_ymodel, alpha=0.5, label=f'Sample {i+1}')

# Plot the synthetic data or observation x for comparison
ax.plot(time, x_star.numpy(), color="red", linestyle='--', label="Observation x", linewidth=2)

ax.set_xlabel("Time [arbitrary units]")
ax.set_ylabel("Flux [arbitrary units]")
ax.set_title(f'Peak Plots of Posterior Samples (N={ncomp_star})')
ax.set_xlim(time[0], time[-1])
ax.set_ylim(0, np.max(x_star.numpy()) * 1.5)  # Adjust y-axis limit for better visualization
ax.legend()
fig.tight_layout()
fig.savefig('experiments/plots_test/peak_plots.png')
plt.close(fig)