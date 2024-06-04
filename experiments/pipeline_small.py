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
fixed_theta = torch.from_numpy(burstparams).float()
fixed_x = torch.from_numpy(ycounts).float()
num_target = torch.tensor([ncomp - 1], dtype=torch.long)  # Adjust the target to 0-based index

# Ensure theta has the maximum theta dimension by padding
pad_size = 4 * max_ncomp - fixed_theta.size(0)
if pad_size > 0:
    fixed_theta = F.pad(fixed_theta, (0, pad_size), "constant", 0)

# Training loop for overfitting
estimator.train()
num_estimator.train()

overfit_loss_values = []
start_time = time_module.time()

for epoch in range(10000):
    optimizer.zero_grad()  # Clear gradients for the estimator
    num_optimizer.zero_grad()  # Clear gradients for the categorical model
    
    loss_value = loss(fixed_theta, fixed_x)
    num_loss_value = categorical_loss(num_estimator(fixed_x.unsqueeze(0)), num_target)  # Ensure x is batched
    
    # Backpropagate the losses
    loss_value.backward()
    num_loss_value.backward()
    
    # Step the optimizers
    optimizer.step()
    num_optimizer.step()
    
    overfit_loss_values.append(loss_value.item())
    if epoch % 1000 == 0:
        print(f"Overfitting Epoch {epoch+1}, Loss: {loss_value.item()}, Categorical Loss: {num_loss_value.item()}")

training_time = time_module.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Evaluation after overfitting
estimator.eval()
num_estimator.eval()

with torch.no_grad():
    samples_theta_given_x_N = estimator.flow(fixed_x).sample((1000,))
    mean_sample = samples_theta_given_x_N.mean(0)
    print("Fixed Input theta:", fixed_theta)
    print("Mean of sampled thetas from posterior after overfitting:", mean_sample)

    # Plotting the samples using corner
    figure = corner.corner(samples_theta_given_x_N.numpy(), labels=[f"Param {i+1}" for i in range(samples_theta_given_x_N.size(1))],
                           truths=fixed_theta.numpy(), title="Posterior Samples vs Fixed Input Theta")
    figure.savefig('experiments/plots_small/posterior_samples_fixed_theta_corner.png')
    plt.close(figure)

    # Plotting the overfitting loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(overfit_loss_values, label='Overfitting Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Overfitting Loss Curve')
    plt.legend()
    plt.savefig('experiments/plots_small/overfitting_loss_curve.png')
    plt.close()


