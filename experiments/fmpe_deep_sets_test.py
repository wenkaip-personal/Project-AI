import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import zuko

from itertools import islice
from lampe.data import JointLoader
from lampe.inference import FMPE, FMPELoss
from lampe.plots import corner, mark_point, nice_rc
from lampe.utils import GDStep
from tqdm import trange

# For DeepSet FMPE
from fmpe_deep_sets import DeepSetFMPE, DeepSetFMPELoss

# Define the simulator function
def simulator(theta: torch.Tensor) -> torch.Tensor:
    # Function f(theta1, theta2) = theta1 ** 2 + theta2 ** 2
    f_val = theta[..., 0] ** 2 + theta[..., 1] ** 2
    # Additive noise version: f_val + noise
    noise = 0.05 * torch.randn_like(f_val)
    return f_val + noise  # Ensure x is 1D

# Define the prior distribution
LOWER = -2 * torch.ones(2)
UPPER = 2 * torch.ones(2)
prior = zuko.distributions.BoxUniform(LOWER, UPPER)

# Define the JointLoader
loader = JointLoader(prior, simulator, batch_size=256, vectorized=True)

# Basic FMPE
estimator_basic = FMPE(2, 1)
loss_basic = FMPELoss(estimator_basic)

# DeepSet FMPE
estimator_deepset = DeepSetFMPE(2, 1)
loss_deepset = DeepSetFMPELoss(estimator_deepset)

# Training routine
def train(estimator, loss, loader, epochs=128, batches_per_epoch=256):
    optimizer = optim.Adam(estimator.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

    estimator.train()

    for epoch in (bar := trange(epochs, unit='epoch')):
        losses = []

        for theta, x in islice(loader, batches_per_epoch):  # batches per epoch
            # Assuming x needs to be unsqueezed to match theta's dimensions if not already done
            x = x.unsqueeze(-1) if x.dim() == 1 else x

            loss_value = loss(theta, x)
            losses.append(step(loss_value))

        bar.set_postfix(loss=torch.stack(losses).mean().item())

# Train both estimators
train(estimator_basic, loss_basic, loader)
train(estimator_deepset, loss_deepset, loader)

# Example inference
theta_star = prior.sample()
x_star = simulator(theta_star)

# Ensure tensors are at least 1-dimensional
if theta_star.dim() == 0:
    theta_star = theta_star.unsqueeze(-1)
if x_star.dim() == 0:
    x_star = x_star.unsqueeze(-1)

# Ensure x_star and theta_star are correctly shaped
x_star = x_star.unsqueeze(0) if x_star.dim() == 1 else x_star
theta_star = theta_star.unsqueeze(0) if theta_star.dim() == 1 else theta_star

estimator_basic.eval()
estimator_deepset.eval()

with torch.no_grad():
    log_p_basic = estimator_basic.flow(x_star).log_prob(theta_star)
    samples_basic = estimator_basic.flow(x_star).sample((2**14,))
    
    log_p_deepset = estimator_deepset.flow(x_star).log_prob(theta_star)
    samples_deepset = estimator_deepset.flow(x_star).sample((2**14,))

print(log_p_basic)
print(log_p_deepset)

# Set x_star to 1 for the specific case where f(theta1, theta2) = 1
x_star = torch.tensor([[1.0]])

# Example inference with x_star set to 1
theta_star = prior.sample()  # Sampling theta_star from the prior for logging purposes

# Ensure tensors are at least 1-dimensional
if theta_star.dim() == 0:
    theta_star = theta_star.unsqueeze(-1)
if x_star.dim() == 0:
    x_star = x_star.unsqueeze(-1)

# Ensure x_star and theta_star are correctly shaped
x_star = x_star.unsqueeze(0) if x_star.dim() == 1 else x_star
theta_star = theta_star.unsqueeze(0) if theta_star.dim() == 1 else theta_star

estimator_basic.eval()
estimator_deepset.eval()

with torch.no_grad():
    log_p_basic = estimator_basic.flow(x_star).log_prob(theta_star)
    samples_basic = estimator_basic.flow(x_star).sample((2**14,))
    
    log_p_deepset = estimator_deepset.flow(x_star).log_prob(theta_star)
    samples_deepset = estimator_deepset.flow(x_star).sample((2**14,))

print(log_p_basic)
print(log_p_deepset)