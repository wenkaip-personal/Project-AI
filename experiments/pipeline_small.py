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

# Initialize models, loss functions, and optimizers
estimator = DeepSetFMPE(theta_dim=4 * max_ncomp, x_dim=1000, freqs=5)
loss = DeepSetFMPELoss(estimator)
optimizer = optim.Adam(estimator.parameters(), lr=1e-3)
step = GDStep(optimizer, clip=1.0)

num_estimator = CategoricalModel(x_dim=1000, max_components=max_ncomp)
num_optimizer = optim.Adam(num_estimator.parameters(), lr=1e-3)
num_step = GDStep(num_optimizer, clip=1.0)

# Generate a single sample for overfitting
ncomp = 2
burstparams = generate_burst_params(ncomp)
ymodel, ycounts = simulate_burst(time, ncomp, burstparams, ybkg*10, return_model=True, noise_type='gaussian')
theta = torch.from_numpy(burstparams).float()
x = torch.from_numpy(ycounts).float()
num_target = torch.tensor([ncomp - 1], dtype=torch.long)  # Adjust the target to 0-based index

# Ensure theta has the maximum theta dimension by padding
pad_size = 4 * max_ncomp - theta.size(0)
if pad_size > 0:
    theta = F.pad(theta, (0, pad_size), "constant", 0)

# Training loop
estimator.train()
num_estimator.train()

loss_values = []
start_time = time_module.time()

for epoch in range(128):
    loss_value = step(loss(theta, x))
    num_loss_value = num_step(categorical_loss(num_estimator(x.unsqueeze(0)), num_target))  # Ensure x is batched
    loss_values.append(loss_value.item())
    print(f"Epoch {epoch+1}, Loss: {loss_value.item()}, Categorical Loss: {num_loss_value.item()}")

training_time = time_module.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Evaluation
estimator.eval()
num_estimator.eval()

with torch.no_grad():
    samples_theta_given_x_N = estimator.flow(x).sample((1000,))
    mean_sample = samples_theta_given_x_N.mean(0)
    print("Input theta:", theta)
    print("Mean of sampled thetas from posterior:", mean_sample)

    # Plotting the samples using corner
    figure = corner.corner(samples_theta_given_x_N.numpy(), labels=[f"Param {i+1}" for i in range(samples_theta_given_x_N.size(1))],
                           truths=theta.numpy(), title="Posterior Samples vs Input Theta")
    figure.savefig('experiments/plots_small/posterior_samples_corner.png')
    plt.close(figure)

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('experiments/plots_small/loss_curve.png')
    plt.close()

