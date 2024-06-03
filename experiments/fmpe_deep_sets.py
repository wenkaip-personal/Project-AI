import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

class DeepSetPhi(nn.Module):
    """
    DeepSet's Phi network for processing individual elements.
    
    This network acts on individual elements of a set, transforming each element
    from the input dimension to an intermediate representation. The transformation
    is performed by a simple feedforward neural network with two linear layers,
    each followed by a ReLU activation function. The output of this network is
    then used by the DeepSetRho network for aggregation.
    
    Attributes:
        network (nn.Sequential): A sequential container of two linear layers and ReLU
                                 activations. Transforms input features to an intermediate
                                 representation.
    
    Args:
        input_dim (int): The dimensionality of the input features for each element.
        output_dim (int): The dimensionality of the output features for each element.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

class DeepSetRho(nn.Module):
    """
    DeepSet's Rho network for aggregating processed elements.
    
    This network aggregates the transformed representations of the set elements
    produced by the DeepSetPhi network. It operates on the aggregated representation
    to produce a single output vector for the entire set. The aggregation is
    performed by a simple feedforward neural network with one hidden layer.
    
    Attributes:
        network (nn.Sequential): A sequential container of two linear layers and a ReLU
                                 activation between them. Aggregates the intermediate
                                 representations into a single output vector.
    
    Args:
        input_dim (int): The dimensionality of the aggregated input features.
        output_dim (int): The dimensionality of the output features for the entire set.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Sum aggregation for permutation invariance
        aggregated = x.sum(dim=1)  # Assuming x has shape [batch_size, set_size, features]
        return self.network(aggregated)

class DeepSetFMPE(nn.Module):
    """
    DeepSet version of the FMPE network.
    
    This class implements the DeepSet approach for the Flow Matching Posterior Estimation
    (FMPE) task. It combines the DeepSetPhi and DeepSetRho networks to process sets of data
    and produce a vector field that induces a time-continuous normalizing flow. This flow
    can be used to transform a distribution into a standard Gaussian distribution.
    
    Attributes:
        phi (DeepSetPhi): The Phi network for processing individual set elements.
        rho (DeepSetRho): The Rho network for aggregating the processed elements.
        freqs (Tensor): A tensor of frequencies used for time embedding.
        zeros (Tensor): A tensor of zeros used as part of the base distribution for the flow.
        ones (Tensor): A tensor of ones used as part of the scale parameter for the base distribution.
    
    Args:
        theta_dim (int): The dimensionality of the parameter space.
        x_dim (int): The dimensionality of the observation space.
        freqs (int, optional): The number of frequencies to use for time embedding.
    """
    def __init__(self, theta_dim: int, x_dim: int, freqs: int):
        super().__init__()
        self.phi = DeepSetPhi(input_dim=theta_dim + x_dim + 2 * freqs, output_dim=128)
        self.rho = DeepSetRho(input_dim=128, output_dim=theta_dim)
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)
        self.register_buffer('zeros', torch.zeros(theta_dim))
        self.register_buffer('ones', torch.ones(theta_dim))

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass for the DeepSetFMPE model.
        
        This function processes input parameters (theta), observations (x), and time (t)
        to produce a transformed representation suitable for inducing a normalizing flow.
        
        Steps involved in the forward pass:
        1. Time Embedding: The time variable (t) is first expanded to match the batch dimension
           and then embedded using sinusoidal functions (cosine and sine) scaled by predefined
           frequencies. This sinusoidal embedding captures the time-dependent dynamics of the flow.
        
        2. Concatenation: The parameters (theta), observations (x), and the time embeddings are
           concatenated to form a single tensor. This tensor contains all necessary information
           for each set element and is ready to be processed by the Phi network.
        
        3. Phi Network Processing: The concatenated tensor is passed through the Phi network
           (DeepSetPhi), which processes each element individually. The Phi network transforms
           each element from its original dimensionality to an intermediate representation.
        
        4. Rho Network Aggregation: The output of the Phi network is directly passed to the Rho network
           (DeepSetRho). The Rho network operates on the entire set of transformed representations to
           produce a single output vector for the entire set. This vector represents the transformed parameters
           at the given time.
        
        Args:
            theta (Tensor): The parameters with shape (N, D), where N is the batch size and
                            D is the dimensionality of the parameter space.
            x (Tensor): The observations with shape (N, L), where L is the dimensionality of
                        the observation space.
            t (Tensor): The time variable with shape (N,), representing the time at which the
                        transformation is evaluated.
        
        Returns:
            Tensor: The output vector for the entire set, representing the transformed
                    parameters at time t.
        """
        t = t.unsqueeze(-1)
        t_cos = torch.cos(t * self.freqs)
        t_sin = torch.sin(t * self.freqs)
        t_embedded = torch.cat((t_cos, t_sin), dim=-1)
        theta, x, t_embedded = broadcast(theta, x, t_embedded, ignore=1)
        input_tensor = torch.cat((theta, x, t_embedded), dim=-1)
        phi_output = self.phi(input_tensor)
        final_output = self.rho(phi_output.unsqueeze(1))
        return final_output

    def flow(self, x: Tensor) -> Distribution:
        """
        Constructs a normalizing flow using the DeepSetFMPE model.
        
        This function creates a normalizing flow that can transform a base distribution
        (in this case, a diagonal normal distribution) into another distribution as defined
        by the DeepSetFMPE model. The transformation is parameterized by the output of the
        DeepSetFMPE model, which acts as the vector field inducing the flow.
        
        The flow is defined over a continuous range of time from t0 to t1 (0 to 1 in this
        implementation). The transformation function (f) is a lambda function that wraps
        the forward pass of the DeepSetFMPE model, allowing it to be used with any time
        input within the specified range.
        
        Args:
            x (Tensor): The observations with shape (N, L), where L is the dimensionality of
                        the observation space.
        
        Returns:
            Distribution: A NormalizingFlow object representing the transformed distribution.
                          This object can be used to evaluate the density or generate samples
                          from the transformed distribution.
        """
        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=lambda t, theta: self(theta, x, t),
                t0=x.new_tensor(0.0),
                t1=x.new_tensor(1.0),
                phi=(x, *self.parameters()),
            ),
            base=DiagNormal(self.zeros, self.ones).expand(x.shape[:-1]),
        )

class DeepSetFMPELoss(nn.Module):
    """
    Module that calculates the flow matching loss for a DeepSetFMPE regressor.
    
    This loss function is designed for training the DeepSetFMPE network. It calculates
    the flow matching loss, which measures the discrepancy between the estimated vector
    field and the true vector field that transforms the posterior distribution into a
    standard Gaussian distribution.
    
    Attributes:
        estimator (DeepSetFMPE): The DeepSetFMPE network being trained.
        eta (float): A small constant used to ensure numerical stability.
    
    Args:
        estimator (DeepSetFMPE): The DeepSetFMPE network to calculate the loss for.
        eta (float, optional): A small constant added to the transformation for numerical stability.
    """
    def __init__(self, estimator: DeepSetFMPE, eta: float = 1e-3):
        super().__init__()
        self.estimator = estimator
        self.eta = eta

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        """
        Calculates the flow matching loss for the DeepSetFMPE model.
        
        This function computes the loss by first sampling a random time (t) for each
        parameter-observation pair in the batch. It then perturbs the parameters (theta)
        linearly between their original value and a random noise vector (epsilon), scaled
        by the sampled time and a small constant (eta) for numerical stability.
        
        The perturbed parameters, along with the original observations and the sampled time,
        are passed through the DeepSetFMPE model to estimate the vector field (v). The loss
        is the mean squared error between this estimated vector field and the target vector
        field, which is the difference between the noise vector and the original parameters.
        
        Args:
            theta (Tensor): The parameters with shape (N, D), where N is the batch size and
                            D is the dimensionality of the parameter space.
            x (Tensor): The observations with shape (N, L), where L is the dimensionality of
                        the observation space.
        
        Returns:
            Tensor: The scalar loss representing the discrepancy between the estimated and
                    true vector fields.
        """
        t = torch.rand(theta.shape[:-1], dtype=theta.dtype, device=theta.device)
        t_ = t[..., None]

        eps = torch.randn_like(theta)
        theta_prime = (1 - t_) * theta + (t_ + self.eta) * eps
        v = eps - theta

        return (self.estimator(theta_prime, x, t) - v).square().mean()


