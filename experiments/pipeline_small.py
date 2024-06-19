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

# Initialize models, loss functions, and optimizers
estimator = FMPE(theta_dim=1 * max_ncomp, x_dim=1000, freqs=5).to(device)
loss = FMPELoss(estimator)
optimizer = optim.Adam(estimator.parameters(), lr=1e-4)
step = GDStep(optimizer, clip=1.0)

num_estimator = CategoricalModel(x_dim=1000, max_components=max_ncomp).to(device)
num_optimizer = optim.Adam(num_estimator.parameters(), lr=1e-4)
num_step = GDStep(num_optimizer, clip=1.0)

# Generate a single sample for overfitting
ncomp = 2
burstparams = generate_burst_params(ncomp)
ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg*10, return_model=True, noise_type='gaussian')
fixed_t0 = torch.from_numpy(burstparams[:ncomp]).float().to(device)  # Only t0
fixed_x = torch.from_numpy(ycounts).float().to(device)

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

# Increase the number of epochs for more extensive training
num_epochs = 50000

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear gradients for the estimator
    
    loss_value = loss(fixed_t0_batch, fixed_x_fft_batch)
    
    # Backpropagate the loss
    loss_value.backward()
    
    # Step the optimizer
    optimizer.step()
    
    overfit_loss_values.append(loss_value.item())
    if epoch % 1000 == 0:
        print(f"Overfitting Epoch {epoch+1}, Loss: {loss_value.item()}")

training_time = time_module.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Evaluation after overfitting
estimator.eval()

with torch.no_grad():
    # Sampling a large number of t0s for each example in the batch to assess overfitting
    num_samples = 1000
    samples_t0_given_x_N = estimator.flow(fixed_x_fft_batch).sample((num_samples,))
    
    # Calculating the mean and standard deviation of sampled t0s across the samples dimension
    mean_sample = samples_t0_given_x_N.mean(dim=0)
    std_sample = samples_t0_given_x_N.std(dim=0)
    
    # Calculating the batch mean and standard deviation of sampled t0s across the batch dimension
    batch_mean_sample = samples_t0_given_x_N.mean(dim=[0, 1])
    batch_std_sample = samples_t0_given_x_N.std(dim=[0, 1])
    
    print("Fixed Input t0:", fixed_t0_batch[0])
    print("Mean of sampled t0s from posterior after overfitting:", mean_sample)
    print("Standard deviation of sampled t0s from posterior after overfitting:", std_sample)
    print("Batch mean of sampled t0s from posterior after overfitting:", batch_mean_sample)
    print("Batch standard deviation of sampled t0s from posterior after overfitting:", batch_std_sample)

    # Plotting the samples using corner
    figure = corner.corner(samples_t0_given_x_N.view(-1, samples_t0_given_x_N.size(-1)).cpu().numpy(), 
                           labels=[f"t0_{i}" for i in range(samples_t0_given_x_N.size(-1))],
                           truths=fixed_t0_batch[0].cpu().numpy(), title="Posterior Samples vs Fixed Input t0")
    figure.savefig('plots_small/posterior_samples_fixed_t0_corner_fft_overfit.png')
    plt.close(figure)

    # Plotting the overfitting loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(overfit_loss_values, label='Overfitting Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Overfitting Loss Curve')
    plt.legend()
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


