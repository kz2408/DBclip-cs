"""
Main entry point for the ResampleLoss sensitivity analysis framework.

Usage examples
--------------

Scan a single parameter (using the built-in simulated evaluator):

.. code-block:: bash

    python tools/sensitivity_analysis/run_sensitivity_analysis.py \\
        --mode single --param focal.balance_param

Scan all parameters and generate overview plots / reports:

.. code-block:: bash

    python tools/sensitivity_analysis/run_sensitivity_analysis.py \\
        --mode all --output_dir ./sensitivity_results

Grid search over two parameters:

.. code-block:: bash

    python tools/sensitivity_analysis/run_sensitivity_analysis.py \\
        --mode multi \\
        --params focal.balance_param focal.gamma \\
        --max_combinations 20

Random search across all parameters:

.. code-block:: bash

    python tools/sensitivity_analysis/run_sensitivity_analysis.py \\
        --mode random --n_random 50

Use the real ResampleLoss (requires a GPU and a class-frequency file):

.. code-block:: bash

    python tools/sensitivity_analysis/run_sensitivity_analysis.py \\
        --mode all \\
        --use_real_loss \\
        --freq_file appendix/coco/longtail2017/class_freq.pkl
"""

import argparse
import os
import sys

import numpy as np

# Allow running from the repository root without installing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.sensitivity_analysis.param_grid import ParamGrid, DEFAULT_CONFIG
from tools.sensitivity_analysis.single_param_scan import SingleParamScanner
from tools.sensitivity_analysis.multi_param_test import MultiParamTester
from tools.sensitivity_analysis.visualization import SensitivityVisualizer
from tools.sensitivity_analysis.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Evaluator factories
# ---------------------------------------------------------------------------

def create_real_evaluator(freq_file, n_samples=32, n_classes=80, use_cuda=True):
    """Build an evaluator that wraps the actual :class:`ResampleLoss`.

    The evaluator creates a random batch of classification logits and
    multi-hot labels, runs a forward pass, and returns the scalar loss.

    Args:
        freq_file (str): Path to the ``class_freq.pkl`` file expected by
            :class:`ResampleLoss`.
        n_samples (int): Batch size used for the forward pass.
        n_classes (int): Number of classes.
        use_cuda (bool): Move tensors to CUDA if ``True``.

    Returns:
        callable: ``evaluator(config) -> {'loss': float}``
    """
    import torch
    from mllt.models.losses.resample_loss import ResampleLoss

    def evaluator(config):
        loss_config = {
            'use_sigmoid': config.get('use_sigmoid', True),
            'reduction': 'mean',
            'loss_weight': 1.0,
            'partial': False,
            'focal': config.get('focal', {'focal': True, 'balance_param': 2.0, 'gamma': 2}),
            'CB_loss': config.get('CB_loss', {'CB_beta': 0.9, 'CB_mode': 'average_w'}),
            'map_param': config.get('map_param', {'alpha': 0.1, 'beta': 10.0, 'gamma': 0.3}),
            'logit_reg': config.get('logit_reg', {'neg_scale': 5.0, 'init_bias': 0.05}),
            'reweight_func': config.get('reweight_func', 'rebalance'),
            'weight_norm': config.get('weight_norm', None),
            'freq_file': freq_file,
        }

        criterion = ResampleLoss(**loss_config)

        cls_score = torch.randn(n_samples, n_classes)
        label = (torch.rand(n_samples, n_classes) > 0.8).long()
        if use_cuda:
            cls_score = cls_score.cuda()
            label = label.cuda()

        with torch.no_grad():
            loss = criterion(cls_score, label)

        loss_val = loss.item() if loss.numel() == 1 else loss.mean().item()
        return {'loss': float(loss_val)}

    return evaluator


def create_simulated_evaluator():
    """Build a lightweight evaluator that *simulates* ResampleLoss behaviour.

    This evaluator requires neither a GPU nor a class-frequency file.  The
    loss value is a deterministic function of the parameter configuration
    (plus a small reproducible noise term) designed to mimic the qualitative
    behaviour of ResampleLoss around its optimal operating point.

    Returns:
        callable: ``evaluator(config) -> {'loss': float, 'simulated_mAP': float}``
    """
    rng = np.random.RandomState(42)

    def evaluator(config):
        focal = config.get('focal', {})
        logit_reg = config.get('logit_reg', {})
        map_param = config.get('map_param', {})

        balance_param = float(focal.get('balance_param', 2.0))
        gamma = float(focal.get('gamma', 2))
        init_bias = float(logit_reg.get('init_bias', 0.05))
        neg_scale = float(logit_reg.get('neg_scale', 5.0))
        alpha = float(map_param.get('alpha', 0.1))
        beta = float(map_param.get('beta', 10.0))
        map_gamma = float(map_param.get('gamma', 0.3))

        # Simulate individual parameter effects (Gaussian wells around optimum)
        focal_effect = (
            np.exp(-0.15 * (balance_param - 2.0) ** 2)
            * np.exp(-0.08 * (gamma - 2.0) ** 2)
        )
        reg_effect = (
            np.exp(-3.0 * (init_bias - 0.05) ** 2)
            * np.exp(-0.03 * (neg_scale - 5.0) ** 2)
        )
        map_effect = (
            np.exp(-5.0 * (alpha - 0.1) ** 2)
            * np.exp(-0.005 * (beta - 10.0) ** 2)
            * np.exp(-2.0 * (map_gamma - 0.3) ** 2)
        )

        # Weight applied to each grouped parameter effect (controls sensitivity range)
        _EFFECT_WEIGHT = 0.25

        # Combined loss (higher effect → lower loss)
        base_loss = 2.5
        loss = (
            base_loss
            * (1.0 - _EFFECT_WEIGHT * focal_effect)
            * (1.0 - _EFFECT_WEIGHT * reg_effect)
            * (1.0 - _EFFECT_WEIGHT * map_effect)
        )

        # Add small reproducible noise
        loss += rng.normal(0, 0.02)
        loss = max(loss, 0.01)

        # Simulated mAP (inversely correlated with loss)
        simulated_map = max(0.0, min(1.0, 1.0 - loss / 5.0))

        return {
            'loss': round(float(loss), 6),
            'simulated_mAP': round(float(simulated_map), 6),
        }

    return evaluator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='ResampleLoss Parameter Sensitivity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'multi', 'random', 'all'],
        default='all',
        help='Analysis mode (default: all)',
    )
    parser.add_argument(
        '--param',
        type=str,
        default='focal.balance_param',
        help='Parameter to scan in single mode (default: focal.balance_param)',
    )
    parser.add_argument(
        '--params',
        nargs='+',
        default=['focal.balance_param', 'focal.gamma'],
        help='Parameters for multi / random mode',
    )
    parser.add_argument(
        '--n_random',
        type=int,
        default=50,
        help='Number of random configurations (random mode, default: 50)',
    )
    parser.add_argument(
        '--max_combinations',
        type=int,
        default=None,
        help='Cap on factorial combinations for grid search',
    )
    parser.add_argument(
        '--freq_file',
        type=str,
        default=None,
        help='Path to class_freq.pkl (required for --use_real_loss)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./sensitivity_results',
        help='Output directory (default: ./sensitivity_results)',
    )
    parser.add_argument(
        '--use_real_loss',
        action='store_true',
        help='Use actual ResampleLoss instead of the simulated evaluator '
             '(requires --freq_file and a CUDA-capable GPU)',
    )
    parser.add_argument(
        '--show_plots',
        action='store_true',
        help='Display plots interactively (requires a graphical display)',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='loss',
        help='Primary metric for analysis and ranking (default: loss)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("ResampleLoss Sensitivity Analysis Framework")
    print("=" * 60)

    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Parameter grid -----------------------------------------------
    param_grid = ParamGrid()

    # ---- Evaluator ----------------------------------------------------
    if args.use_real_loss:
        if not args.freq_file:
            raise ValueError("--freq_file is required when --use_real_loss is set.")
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False
        print(f"Using real ResampleLoss evaluator (CUDA={use_cuda}) ...")
        evaluator = create_real_evaluator(
            freq_file=args.freq_file,
            use_cuda=use_cuda,
        )
    else:
        print("Using simulated evaluator (no GPU or freq_file required) ...")
        evaluator = create_simulated_evaluator()

    # ---- Tools --------------------------------------------------------
    scanner = SingleParamScanner(param_grid, evaluator, output_dir=args.output_dir)
    multi_tester = MultiParamTester(param_grid, evaluator, output_dir=args.output_dir)
    visualizer = SensitivityVisualizer(output_dir=plots_dir)
    reporter = ReportGenerator(output_dir=args.output_dir)

    all_scan_results = {}
    multi_param_results = []

    # ---- Run analysis -------------------------------------------------
    if args.mode == 'single':
        result = scanner.scan(args.param)
        all_scan_results[args.param] = result
        visualizer.plot_single_param_scan(result, show=args.show_plots)

    elif args.mode == 'multi':
        multi_param_results = multi_tester.grid_search(
            args.params, max_combinations=args.max_combinations
        )
        if len(args.params) == 2:
            visualizer.plot_heatmap(
                multi_param_results, args.params,
                metric_key=args.metric, show=args.show_plots,
            )

    elif args.mode == 'random':
        multi_param_results = multi_tester.random_search(
            args.n_random, param_names=args.params if args.params else None
        )

    else:  # 'all'
        print("\nRunning single-parameter scans for all parameters ...")
        all_scan_results = scanner.scan_all()

        visualizer.plot_all_scans(all_scan_results, show=args.show_plots)
        visualizer.plot_sensitivity_comparison(
            all_scan_results, metric_key=args.metric, show=args.show_plots
        )

        print("\nRunning grid search over focal parameters ...")
        focal_results = multi_tester.grid_search(
            ['focal.balance_param', 'focal.gamma'],
            max_combinations=args.max_combinations,
        )
        multi_param_results.extend(focal_results)
        visualizer.plot_heatmap(
            focal_results, ['focal.balance_param', 'focal.gamma'],
            metric_key=args.metric, show=args.show_plots,
        )

        print("\nRunning grid search over map_param parameters ...")
        map_results = multi_tester.grid_search(
            ['map_param.alpha', 'map_param.beta'],
            max_combinations=args.max_combinations,
        )
        multi_param_results.extend(map_results)
        visualizer.plot_heatmap(
            map_results, ['map_param.alpha', 'map_param.beta'],
            metric_key=args.metric, show=args.show_plots,
        )

    # ---- Reports ------------------------------------------------------
    print("\nGenerating reports ...")
    if all_scan_results:
        reporter.generate_scan_report(all_scan_results, metric_key=args.metric)
    if multi_param_results:
        reporter.generate_multi_param_report(multi_param_results, metric_key=args.metric)
    reporter.generate_json_summary(all_scan_results, multi_param_results)

    print(f"\nAnalysis complete!  Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
