"""
Parameter search grid definition module for ResampleLoss sensitivity analysis.

Defines default parameter ranges and provides utilities for generating
parameter configurations for sensitivity analysis experiments.
"""

import copy
import numpy as np
from itertools import product


# Default baseline configuration matching the problem statement
DEFAULT_CONFIG = {
    'use_sigmoid': True,
    'reweight_func': 'rebalance',
    'focal': {
        'focal': True,
        'balance_param': 2.0,
        'gamma': 2,
    },
    'logit_reg': {
        'neg_scale': 5.0,
        'init_bias': 0.05,
    },
    'map_param': {
        'alpha': 0.1,
        'beta': 10.0,
        'gamma': 0.3,
    },
    'CB_loss': {
        'CB_beta': 0.9,
        'CB_mode': 'average_w',
    },
    'weight_norm': None,
}

# Default parameter ranges for sensitivity analysis
DEFAULT_PARAM_RANGES = {
    'focal.balance_param': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    'focal.gamma': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    'logit_reg.init_bias': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    'logit_reg.neg_scale': [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    'map_param.alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'map_param.beta': [1.0, 2.0, 5.0, 10.0, 15.0, 20.0],
    'map_param.gamma': [0.1, 0.2, 0.3, 0.4, 0.5],
    'use_sigmoid': [True, False],
    'reweight_func': [None, 'inv', 'sqrt_inv', 'rebalance', 'CB'],
}


class ParamGrid:
    """Parameter search grid generator for ResampleLoss sensitivity analysis.

    This class manages the parameter space for sensitivity analysis experiments.
    It supports generating single-parameter scans, full factorial grids, and
    random samples from the parameter space.

    Example usage::

        grid = ParamGrid()
        # Scan a single parameter
        configs = grid.get_single_param_configs('focal.balance_param')
        # Grid search over two parameters
        configs = grid.get_grid_configs(['focal.gamma', 'map_param.alpha'])
        # Random sample
        configs = grid.get_random_configs(n_configs=20)
    """

    def __init__(self, base_config=None, param_ranges=None):
        """Initialize the parameter grid.

        Args:
            base_config (dict, optional): Base configuration for ResampleLoss.
                Defaults to DEFAULT_CONFIG.
            param_ranges (dict, optional): Parameter ranges for sensitivity
                analysis. Keys are dotted parameter names (e.g.,
                ``'focal.balance_param'``), values are lists of candidate
                values. Defaults to DEFAULT_PARAM_RANGES.
        """
        self.base_config = base_config or copy.deepcopy(DEFAULT_CONFIG)
        self.param_ranges = param_ranges or copy.deepcopy(DEFAULT_PARAM_RANGES)

    def get_single_param_configs(self, param_name):
        """Generate configurations for a single-parameter scan.

        All parameters except ``param_name`` are held at their baseline values.

        Args:
            param_name (str): Dotted parameter name to scan, e.g.
                ``'focal.balance_param'``.

        Returns:
            list[tuple]: List of ``(config_dict, param_value)`` pairs, one
                entry per candidate value in ``self.param_ranges[param_name]``.

        Raises:
            ValueError: If ``param_name`` is not found in ``param_ranges``.
        """
        if param_name not in self.param_ranges:
            raise ValueError(
                f"Parameter '{param_name}' not found in param_ranges. "
                f"Available: {list(self.param_ranges.keys())}"
            )

        configs = []
        for value in self.param_ranges[param_name]:
            config = copy.deepcopy(self.base_config)
            config = self._set_param(config, param_name, value)
            configs.append((config, value))

        return configs

    def get_grid_configs(self, param_names, max_combinations=None):
        """Generate configurations for a full factorial grid search.

        Args:
            param_names (list[str]): Dotted parameter names to vary.
            max_combinations (int, optional): Maximum number of combinations
                to return. When the full grid exceeds this limit, combinations
                are drawn via random sampling without replacement.

        Returns:
            list[tuple]: List of ``(config_dict, param_dict)`` pairs.

        Raises:
            ValueError: If any name in ``param_names`` is not in
                ``param_ranges``.
        """
        param_value_lists = []
        for name in param_names:
            if name not in self.param_ranges:
                raise ValueError(
                    f"Parameter '{name}' not found in param_ranges. "
                    f"Available: {list(self.param_ranges.keys())}"
                )
            param_value_lists.append(self.param_ranges[name])

        all_combinations = list(product(*param_value_lists))

        if max_combinations is not None and len(all_combinations) > max_combinations:
            indices = np.random.choice(
                len(all_combinations), max_combinations, replace=False
            )
            all_combinations = [all_combinations[i] for i in indices]

        configs = []
        for combination in all_combinations:
            config = copy.deepcopy(self.base_config)
            param_dict = {}
            for name, value in zip(param_names, combination):
                config = self._set_param(config, name, value)
                param_dict[name] = value
            configs.append((config, param_dict))

        return configs

    def get_random_configs(self, n_configs, param_names=None):
        """Generate random parameter configurations.

        Each configuration independently samples one value per parameter from
        the corresponding range.

        Args:
            n_configs (int): Number of random configurations to generate.
            param_names (list[str], optional): Parameters to randomize.
                Defaults to all parameters in ``param_ranges``.

        Returns:
            list[tuple]: List of ``(config_dict, param_dict)`` pairs.
        """
        if param_names is None:
            param_names = list(self.param_ranges.keys())

        configs = []
        for _ in range(n_configs):
            config = copy.deepcopy(self.base_config)
            param_dict = {}
            for name in param_names:
                idx = np.random.randint(len(self.param_ranges[name]))
                value = self.param_ranges[name][idx]
                config = self._set_param(config, name, value)
                param_dict[name] = value
            configs.append((config, param_dict))

        return configs

    def add_param_range(self, param_name, values):
        """Add or update a parameter range.

        Args:
            param_name (str): Dotted parameter name.
            values (list): Candidate values for this parameter.
        """
        self.param_ranges[param_name] = list(values)

    def get_param_names(self):
        """Return all available parameter names.

        Returns:
            list[str]: List of dotted parameter names.
        """
        return list(self.param_ranges.keys())

    def summary(self):
        """Print a human-readable summary of the parameter grid."""
        print("Parameter Grid Summary")
        print("=" * 60)
        print("Base configuration:")
        for key, value in self.base_config.items():
            print(f"  {key}: {value}")
        print("\nParameter ranges:")
        for name, values in self.param_ranges.items():
            print(f"  {name}: {values}")
        print("=" * 60)

    def _set_param(self, config, param_name, value):
        """Set a dotted parameter value in the config dict.

        Supports one or two levels of nesting (e.g. ``'focal.gamma'``).

        Args:
            config (dict): Configuration dict to modify (in-place).
            param_name (str): Dotted parameter name.
            value: New value to assign.

        Returns:
            dict: Modified configuration dict.

        Raises:
            ValueError: If nesting depth is greater than two.
        """
        parts = param_name.split('.')
        if len(parts) == 1:
            config[parts[0]] = value
        elif len(parts) == 2:
            if parts[0] not in config:
                config[parts[0]] = {}
            config[parts[0]][parts[1]] = value
        else:
            raise ValueError(
                f"Nested params with depth > 2 are not supported: {param_name}"
            )
        return config
