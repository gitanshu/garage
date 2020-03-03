"""A value function (baseline) based on a GaussianMLP model."""
import numpy as np
import torch
from torch import nn

from garage.np.baselines import Baseline
from garage.torch.algos import make_optimizer
from garage.torch.modules import GaussianMLPModule


class GaussianMLPBaseline(Baseline):
    """Gaussian MLP Baseline with Model.

    It fits the input data to a gaussian distribution estimated by
    a MLP.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normalize_observation (bool): Bool for normalizing observation or not.
        normalize_return (bool): Bool for normalizing return or not.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in torch.optim.
        optimizer_args (dict): Optimizer arguments.
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normalize_observation=True,
                 normalize_return=True,
                 optimizer=torch.optim.Adam,
                 optimizer_args=None):
        super().__init__(env_spec)

        self._module = GaussianMLPModule(
            input_dim=env_spec.observation_space.flat_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization)

        self._normalize_observation = normalize_observation
        self._normalize_return = normalize_return

        if optimizer_args is None:
            optimizer_args = dict()

        self._optimizer = make_optimizer(optimizer, self._module,
                                         **optimizer_args)

    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (list[dict]): Sample paths.

        """
        # observations = xs, return = ys
        observations = np.concatenate([p['observations'] for p in paths])
        returns = np.concatenate([p['returns'] for p in paths])

        if self._normalize_observation:
            mean = np.mean(observations, axis=0, keepdims=True)
            std = np.std(observations, axis=0, keepdims=True) + 1e-8
            observations = (observations - mean) / std

        if self._normalize_return:
            mean = np.mean(returns, axis=0, keepdims=True)
            std = np.std(returns, axis=0, keepdims=True) + 1e-8
            returns = (returns - mean) / std

        dist = self._module(torch.Tensor(observations))
        ll = dist.log_prob(torch.Tensor(returns))
        loss = -ll.mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def predict(self, path):
        """Predict value based on paths.

        Args:
            path (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
        with torch.no_grad():
            dist = self._module(torch.Tensor(path['observations']))
        return dist.mean.flatten().numpy()

    def get_param_values(self):
        """Get the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Returns:
            dict: The parameters (in the form of the state dictionary).

        """
        return self._module.state_dict()

    def set_param_values(self, flattened_params):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            flattened_params (dict): State dictionary.

        """
        self._module.load_state_dict(flattened_params)
