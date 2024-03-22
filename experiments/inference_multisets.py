import torch
import torch.nn as nn
import math
from lampe.inference import FMPE, FMPELoss
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform

# Dummy build function for compatibility with FMPE's constructor
def dummy_build(*args, **kwargs):
    return nn.Sequential()

class DeepSetFMPE(FMPE):
    """
    Extends the Flow Matching Posterior Estimation (FMPE) network to handle sets of parameters
    and observations in a permutation-invariant manner, leveraging the deep sets principle.
    This implementation processes elements of the sets independently before aggregating their representations,
    applying the FMPE methodology to the aggregated representation.
    """
    def __init__(self, theta_dim, x_dim, freqs=3, hidden_features=64, num_hidden_layers=5, out_features=None, activation=nn.ReLU):
        super().__init__(theta_dim, x_dim, freqs=freqs, build=dummy_build)
        
        self.activation = activation()
        self.hidden_features = hidden_features
        self.phi_theta = self._build_phi_network(theta_dim, hidden_features, num_hidden_layers)
        self.phi_x = self._build_phi_network(x_dim, hidden_features, num_hidden_layers)
        self.time_embedding = nn.Linear(2 * freqs, hidden_features)
        self.rho = nn.Sequential(
            nn.Linear(3 * hidden_features, hidden_features),
            self.activation,
            nn.Linear(hidden_features, theta_dim)
        )
        self.register_buffer('freqs_tensor', torch.arange(1, freqs + 1) * math.pi)

    def _build_phi_network(self, input_dim, hidden_features, num_hidden_layers):
        layers = [nn.Linear(input_dim, hidden_features), self.activation]
        for _ in range(1, num_hidden_layers):
            layers.extend([nn.Linear(hidden_features, hidden_features), self.activation])
        return nn.Sequential(*layers)

    def forward(self, theta, x, t):
        phi_theta = self.phi_theta(theta)
        phi_x = self.phi_x(x)
        if phi_theta.dim() == 1:
            phi_theta = phi_theta.unsqueeze(0)
        if phi_x.dim() == 1:
            phi_x = phi_x.unsqueeze(0)
        
        t = self.freqs_tensor * t.unsqueeze(-1)
        t = torch.cat((t.sin(), t.cos()), dim=-1)
        t_embedded = self.time_embedding(t)
        
        if t_embedded.dim() == 1:
            t_embedded = t_embedded.unsqueeze(0)
        if t_embedded.size(0) != phi_theta.size(0):
            t_embedded = t_embedded.expand(phi_theta.size(0), -1)
        
        aggregated = torch.cat([phi_theta, phi_x, t_embedded], dim=1)
        output = self.rho(aggregated)
        return output.squeeze(0) if output.size(0) == 1 else output

    def flow(self, x):
        """
        Constructs a normalizing flow for a given observation x, integrating the model's
        dynamics and handling gradients for use with the ODE solver.
        """
        def adjusted_f(t, theta):
            with torch.enable_grad():
                theta.requires_grad_(True)
                output = self.forward(theta, x, t)
                grad_outputs = torch.ones_like(output)
                gradients = torch.autograd.grad(output, theta, grad_outputs=grad_outputs, create_graph=True)[0]
            return output, gradients
        
        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=adjusted_f,
                t0=x.new_tensor(0.0),
                t1=x.new_tensor(1.0),
                phi=(x, *self.parameters()),
            ),
            base=DiagNormal(self.zeros, self.ones).expand(x.shape[:-1])
        )

class DeepSetFMPELoss(FMPELoss):
    """
    Defines the loss function for the DeepSetFMPE model, extending FMPELoss. It computes the loss
    for a batch of sets of parameters and observations, handling them in a permutation-invariant manner.
    """
    def __init__(self, estimator: DeepSetFMPE):
        super().__init__(estimator)

    def forward(self, theta, x):
        return super().forward(theta, x)
