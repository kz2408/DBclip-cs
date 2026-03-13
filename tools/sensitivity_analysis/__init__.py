"""
ResampleLoss Parameter Sensitivity Analysis Framework.

This package provides tools for analyzing the sensitivity of ResampleLoss
parameters:
    - focal.balance_param: Focal loss balance parameter
    - focal.gamma: Focal loss exponent
    - logit_reg.init_bias: Initial bias for logit regularization
    - logit_reg.neg_scale: Negative sample scaling
    - map_param.alpha: MAP parameter alpha
    - map_param.beta: MAP parameter beta
    - map_param.gamma: MAP parameter gamma
    - use_sigmoid: Whether to use sigmoid activation
    - reweight_func: Reweighting function

Modules:
    param_grid: Parameter search grid definition
    single_param_scan: Single parameter scan analysis
    multi_param_test: Multi-parameter combination testing
    visualization: Visualization tools
    report_generator: Experiment report generation
"""

from .param_grid import ParamGrid, DEFAULT_CONFIG, DEFAULT_PARAM_RANGES
from .single_param_scan import SingleParamScanner
from .multi_param_test import MultiParamTester
from .visualization import SensitivityVisualizer
from .report_generator import ReportGenerator

__all__ = [
    'ParamGrid',
    'DEFAULT_CONFIG',
    'DEFAULT_PARAM_RANGES',
    'SingleParamScanner',
    'MultiParamTester',
    'SensitivityVisualizer',
    'ReportGenerator',
]
