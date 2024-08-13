import matplotlib.pyplot as plt
import seaborn as sns
import time as time_module
sns.set_style("whitegrid")
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import corner
import numpy as np
from lampe.plots import nice_rc
from lampe.utils import GDStep
from tqdm import trange

from lampe.inference import FMPE, FMPELoss
from fmpe_deep_sets import DeepSetFMPE, DeepSetFMPELoss
from generate_burst_data import simulate_burst

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example usage
time = np.linspace(0, 1.0, 1000)  # Time array 
max_ncomp = 2  # Maximum number of components
ncomp = 1

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

# Improved model architecture
class ImprovedFMPE(FMPE):
    def __init__(self, theta_dim, x_dim, freqs):
        super().__init__(theta_dim, x_dim, freqs)
        input_dim = theta_dim + x_dim + 2 * freqs
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, theta_dim)
        )

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

# Experiment setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
estimator = ImprovedFMPE(theta_dim=1 * ncomp, x_dim=1000, freqs=20).to(device)
loss_fn = FMPELoss(estimator)
# estimator = DeepSetFMPE(theta_dim=1 * ncomp, x_dim=1000, freqs=20, hidden_dim=512).to(device)
# loss_fn = DeepSetFMPELoss(estimator)
optimizer = optim.AdamW(estimator.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.5, verbose=True)
step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

num_estimator = CategoricalModel(x_dim=1000, max_components=max_ncomp).to(device)
num_optimizer = optim.AdamW(num_estimator.parameters(), lr=1e-4, weight_decay=1e-5)
num_scheduler = optim.lr_scheduler.ReduceLROnPlateau(num_optimizer, patience=2000, factor=0.5, verbose=True)
num_step = GDStep(num_optimizer, clip=1.0)

# Training loop
estimator.train()
num_estimator.train()

for epoch in (bar := trange(128, unit='epoch')):
    losses = []
    num_losses = []

    for _ in range(256):  # 256 batches per epoch
        burstparams = generate_burst_params(ncomp)
        ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg*10, return_model=True, noise_type='gaussian')
        t0 = torch.from_numpy(burstparams[:ncomp]).float().to(device)  # Only t0
        x = torch.from_numpy(ycounts).float().to(device)
        num_target = torch.tensor([ncomp - 1], dtype=torch.long).to(device)  # Adjust the target to 0-based index, ensure long type

        # Normalize the input data
        x = (x - x.mean()) / x.std()

        optimizer.zero_grad()
        loss = loss_fn(t0, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(estimator.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        num_optimizer.zero_grad()
        num_loss = categorical_loss(num_estimator(x.unsqueeze(0)), num_target)
        num_loss.backward()
        torch.nn.utils.clip_grad_norm_(num_estimator.parameters(), max_norm=1.0)
        num_optimizer.step()
        num_scheduler.step(num_loss)

        losses.append(loss.item())
        num_losses.append(num_loss.item())

    bar.set_postfix(loss=torch.tensor(losses).mean().item(), num_loss=torch.tensor(num_losses).mean().item())

burstparams_star = generate_burst_params(ncomp)
ymodel_star, ycounts_star = simulate_burst(time, ncomp, burstparams_star, ybkg*10, return_model=True, noise_type='gaussian')
t0_star = torch.from_numpy(burstparams_star[:ncomp]).float().to(device)
x_star = torch.from_numpy(ycounts_star).float().to(device)

# Evaluation
estimator.eval()
num_estimator.eval()

with torch.no_grad():
    log_p_t0_given_x_N = estimator.flow(x_star).log_prob(t0_star)
    samples_t0_given_x_N = estimator.flow(x_star).sample((2**14,))

    p_N_given_x = num_estimator(x_star)

# Compute the joint posterior p(theta, N | x) = p(theta | x, N) * p(N | x)
joint_posterior = torch.exp(log_p_t0_given_x_N) * p_N_given_x

print("Joint posterior p(theta, N | x):")
print(joint_posterior)

# Plotting the samples using corner
if samples_t0_given_x_N.numel() == 0 or torch.isnan(samples_t0_given_x_N).any():
    print("Warning: Samples are empty or contain NaN values. Skipping corner plot.")
else:
    samples_np = samples_t0_given_x_N.cpu().numpy()
    ranges = []
    for i in range(samples_np.shape[1]):
        q1, q3 = np.percentile(samples_np[:, i], [25, 75])
        iqr = q3 - q1
        range_min = max(q1 - 1.5 * iqr, samples_np[:, i].min())
        range_max = min(q3 + 1.5 * iqr, samples_np[:, i].max())
        ranges.append((range_min, range_max))

    figure = corner.corner(
        samples_np,
        labels=[f"t0_{i}" for i in range(samples_np.shape[1])],
        truths=t0_star.cpu().numpy(),
        title="Posterior Samples vs Ground Truth t0",
        levels=(0.68, 0.95, 0.997),
        color='blue',
        truth_color='red',
        show_titles=True,
        title_fmt='.4f',
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.0,
        range=ranges,
        hist_kwargs={'density': True}
    )

    # Adjust the layout to prevent cutting off axis labels
    plt.tight_layout()

    # Save the figure with a higher DPI for better quality
    figure.savefig('experiments/plots/posterior_samples_ground_truth_t0_corner.png', dpi=300, bbox_inches='tight')
    plt.close(figure)

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('experiments/plots/loss_curve.png')
plt.close()

# Plotting the conditioning data (time series)
plt.figure(figsize=(10, 6))
plt.plot(time, ycounts, label='Conditioning Time Series Data')
plt.xlabel('Time')
plt.ylabel('Counts')
plt.title('Conditioning Time Series Data')
plt.legend()
plt.savefig('experiments/plots/conditioning_data_time_series.png')
plt.close()

# Examine the trajectory of samples
trajectory = estimator.flow(x_star).sample((1000,))
plt.figure(figsize=(10, 6))
for i in range(trajectory.shape[1]):
    plt.plot(trajectory[:, i].cpu().numpy(), label=f't0_{i}')
plt.xlabel('Sample')
plt.ylabel('t0 value')
plt.title('Trajectory of Sampled t0s')
plt.legend()
plt.savefig('experiments/plots/trajectory_samples.png')
plt.close()

# Plot the posterior p(N | x)
fig, ax = plt.subplots()
ax.plot(range(1, max_ncomp+1), p_N_given_x.cpu().numpy(), marker='o')
ax.set_xlabel('Number of Components N')
ax.set_ylabel('Posterior Probability p(N | x)')
ax.set_title('Posterior Distribution of N')
fig.tight_layout()
fig.savefig('experiments/plots/posterior_N.png')
plt.close(fig)