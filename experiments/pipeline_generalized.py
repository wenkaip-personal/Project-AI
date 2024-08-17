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

from lampe.inference import FMPE, FMPELoss
from fmpe_deep_sets import DeepSetFMPE, DeepSetFMPELoss
from generate_burst_data import simulate_burst

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example usage
time = np.linspace(0, 1.0, 1000)  # Time array 
max_ncomp = 2  # Maximum number of components
ncomp = 2  # Number of components for this experiment

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(self.net[0].in_features, 512),
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
            nn.Linear(512, self.net[-1].out_features)
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

# Note: We sort the t0 values to ensure consistent ordering of components.
# This is necessary because the order of components in mixture models can be arbitrary.
def sort_t0(t0):
    return torch.sort(t0, dim=-1)[0]

# Experiment setup
# estimator = ImprovedFMPE(theta_dim=1 * ncomp, x_dim=1000, freqs=20).to(device)
# loss_fn = FMPELoss(estimator)
estimator = DeepSetFMPE(theta_dim=1 * ncomp, x_dim=1000, freqs=20, hidden_dim=512).to(device)
loss_fn = DeepSetFMPELoss(estimator)
optimizer = optim.AdamW(estimator.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.5, verbose=True)

num_estimator = CategoricalModel(x_dim=1000, max_components=max_ncomp).to(device)
num_optimizer = optim.Adam(num_estimator.parameters(), lr=1e-4)
num_step = GDStep(num_optimizer, clip=1.0)

# Generate multiple samples in a loop
batch_size = 1024
t0_batch = []
x_batch = []

for i in range(batch_size):
    burstparams = generate_burst_params(ncomp)
    ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg*10, return_model=True, noise_type='gaussian')
    t0 = torch.from_numpy(burstparams[:ncomp]).float().to(device)  # Only t0
    x = torch.from_numpy(ycounts).float().to(device)

    # Modified normalization: preserve relative timing information
    x = (x - x.min()) / (x.max() - x.min())

    t0_batch.append(t0)
    x_batch.append(x)

t0_batch = torch.stack([sort_t0(t0) for t0 in t0_batch])
x_batch = torch.stack(x_batch)

# Training loop
estimator.train()
num_epochs = 100000
loss_values = []
start_time = time_module.time()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_fn(t0_batch, x_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(estimator.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)

    loss_values.append(loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")

    # Early stopping condition
    if loss.item() < 5e-3:
        print(f"Reached desired loss at epoch {epoch}")
        break

training_time = time_module.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

burstparams_star = generate_burst_params(ncomp)
ymodel_star, ycounts_star = simulate_burst(time, ncomp, burstparams_star, ybkg*10, return_model=True, noise_type='gaussian')
t0_star = torch.from_numpy(burstparams_star[:ncomp]).float().to(device)
x_star = torch.from_numpy(ycounts_star).float().to(device)

# Modified normalization for x_star
x_star = (x_star - x_star.min()) / (x_star.max() - x_star.min())

# Evaluation after training
estimator.eval()

with torch.no_grad():
    num_samples = 50000
    sample_batch_size = 1000
    all_samples = []
    
    for i in range(0, num_samples, sample_batch_size):
        batch_samples = estimator.flow(x_star).sample((sample_batch_size,))
        all_samples.append(batch_samples)
    
    samples_t0_given_x_N = torch.cat(all_samples, dim=0)
    
    # Sort the samples
    samples_t0_given_x_N = sort_t0(samples_t0_given_x_N)
    
    mean_sample = samples_t0_given_x_N.mean(dim=0)
    std_sample = samples_t0_given_x_N.std(dim=0)
    
    # Sort the ground truth t0
    t0_star_sorted = sort_t0(t0_star)
    
    print("Ground Truth t0 (sorted):", t0_star_sorted)
    print("Mean of sampled t0s from posterior after training (sorted):", mean_sample)
    print("Standard deviation of sampled t0s from posterior after training:", std_sample)

    # Plotting the samples using corner
    figure = corner.corner(
        samples_t0_given_x_N.cpu().numpy(), 
        labels=[f"t0_{i}" for i in range(samples_t0_given_x_N.size(-1))],
        truths=t0_star_sorted.cpu().numpy(),
        title="Posterior Samples vs Ground Truth t0 (Sorted)",
        levels=(0.68, 0.95, 0.997),
        color='blue',
        truth_color='red',
        show_titles=True,
        title_fmt='.4f',
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.0,
        hist_kwargs={'density': True}
    )

    # Adjust the layout to prevent cutting off axis labels
    plt.tight_layout()

    # Save the figure with a higher DPI for better quality
    figure.savefig('plots_generalized/posterior_samples_ground_truth_t0_corner.png', dpi=300, bbox_inches='tight')
    plt.close(figure)

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('plots_generalized/loss_curve.png')
    plt.close()

    # Plotting the conditioning data (time series)
    plt.figure(figsize=(10, 6))
    plt.plot(time, ycounts_star, label='Conditioning Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    plt.title('Conditioning Time Series Data')
    plt.legend()
    plt.savefig('plots_generalized/conditioning_data_time_series.png')
    plt.close()