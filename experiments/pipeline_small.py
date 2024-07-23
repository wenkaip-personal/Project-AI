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
# from fmpe_deep_sets_small import DeepSetFMPE, DeepSetFMPELoss
from generate_burst_data import simulate_burst

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Experiment setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
estimator = ImprovedFMPE(theta_dim=1 * max_ncomp, x_dim=1000, freqs=20).to(device)
loss_fn = FMPELoss(estimator)
optimizer = optim.AdamW(estimator.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2000, factor=0.5, verbose=True)

num_estimator = CategoricalModel(x_dim=1000, max_components=max_ncomp).to(device)
num_optimizer = optim.Adam(num_estimator.parameters(), lr=1e-4)
num_step = GDStep(num_optimizer, clip=1.0)

# Generate a single sample for overfitting
ncomp = 2
burstparams = generate_burst_params(ncomp)
ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg*10, return_model=True, noise_type='gaussian')
fixed_t0 = torch.from_numpy(burstparams[:ncomp]).float().to(device)  # Only t0
fixed_x = torch.from_numpy(ycounts).float().to(device)

# Normalize the input data
fixed_x = (fixed_x - fixed_x.mean()) / fixed_x.std()

# Transform the time series data using Fourier Transform for better representation
import torch.fft as fft
fixed_x_fft = torch.abs(fft.fft(fixed_x))

# Training loop for overfitting
estimator.train()

overfit_loss_values = []
start_time = time_module.time()

# Generate a large batch of the same single example to improve overfitting
batch_size = 1024
fixed_t0_batch = fixed_t0.repeat(batch_size, 1)
fixed_x_fft_batch = fixed_x_fft.repeat(batch_size, 1)

# Training loop
num_epochs = 100000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_fn(fixed_t0_batch, fixed_x_fft_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(estimator.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)

    overfit_loss_values.append(loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")

    # Early stopping condition
    if loss.item() < 1e-4:
        print(f"Reached desired loss at epoch {epoch}")
        break

training_time = time_module.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Evaluation after overfitting
estimator.eval()

with torch.no_grad():
    # Sampling a large number of t0s for each example in the batch to assess overfitting
    num_samples = 50000
    sample_batch_size = 1000  # Process samples in smaller batches
    all_samples = []
    
    for i in range(0, num_samples, sample_batch_size):
        batch_samples = estimator.flow(fixed_x_fft_batch[:1]).sample((sample_batch_size,))
        all_samples.append(batch_samples)
    
    samples_t0_given_x_N = torch.cat(all_samples, dim=0)
    
    # Calculating the mean and standard deviation of sampled t0s across the samples dimension
    mean_sample = samples_t0_given_x_N.mean(dim=0)
    std_sample = samples_t0_given_x_N.std(dim=0)
    
    print("Fixed Input t0:", fixed_t0_batch[0])
    print("Mean of sampled t0s from posterior after overfitting:", mean_sample)
    print("Standard deviation of sampled t0s from posterior after overfitting:", std_sample)

    # Plotting the samples using corner
    figure = corner.corner(
        samples_t0_given_x_N.cpu().numpy(), 
        labels=[f"t0_{i}" for i in range(samples_t0_given_x_N.size(-1))],
        truths=fixed_t0_batch[0].cpu().numpy(),
        title="Posterior Samples vs Fixed Input t0",
        levels=(0.68, 0.95, 0.997),
        color='blue',
        truth_color='red',
        show_titles=True,
        title_fmt='.4f',
        quantiles=[0.16, 0.5, 0.84],
        smooth=1.0,
        range=[(t0 - 0.01, t0 + 0.01) for t0 in fixed_t0_batch[0].cpu().numpy()],  # Set range to ±0.01 around true values
        hist_kwargs={'density': True}
    )

    # Adjust the layout to prevent cutting off axis labels
    plt.tight_layout()

    # Save the figure with a higher DPI for better quality
    figure.savefig('plots_small/posterior_samples_fixed_t0_corner_fft_overfit.png', dpi=300, bbox_inches='tight')
    plt.close(figure)

    # Plotting the overfitting loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(overfit_loss_values, label='Overfitting Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Overfitting Loss Curve')
    plt.legend()
    plt.yscale('log')
    plt.savefig('plots_small/loss_curve_fft_overfit.png')
    plt.close()

    # Plotting the conditioning data (time series)
    plt.figure(figsize=(10, 6))
    plt.plot(time, ycounts, label='Conditioning Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    plt.title('Conditioning Time Series Data')
    plt.legend()
    plt.savefig('plots_small/conditioning_data_time_series.png')
    plt.close()

    # Examine the trajectory of samples
    trajectory = estimator.flow(fixed_x_fft_batch[0]).sample((1000,))
    plt.figure(figsize=(10, 6))
    for i in range(trajectory.shape[1]):
        plt.plot(trajectory[:, i].cpu().numpy(), label=f't0_{i}')
    plt.xlabel('Sample')
    plt.ylabel('t0 value')
    plt.title('Trajectory of Sampled t0s')
    plt.legend()
    plt.savefig('plots_small/trajectory_samples.png')
    plt.close()