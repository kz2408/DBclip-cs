"""
Multi-parameter combination test tool for ResampleLoss sensitivity analysis.

Provides :class:`MultiParamTester` which supports:

* **Grid search** – exhaustive (or sampled) factorial combinations.
* **Random search** – independently sampled random configurations.
* **Sensitivity analysis** – ranks each parameter by the range of the target
  metric observed when varying it while averaging over all other parameters.
"""

import json
import os
import time
from datetime import datetime

import numpy as np


class MultiParamTester:
    """Test multiple ResampleLoss parameter combinations simultaneously.

    Example usage::

        from tools.sensitivity_analysis import ParamGrid, MultiParamTester

        grid = ParamGrid()
        tester = MultiParamTester(grid, my_evaluator)

        # Full factorial grid over two parameters
        results = tester.grid_search(
            ['focal.balance_param', 'focal.gamma'], max_combinations=20
        )

        # Random search across all parameters
        results = tester.random_search(n_configs=50)

        # Find the best config
        best = tester.get_best_config(metric_key='loss')
    """

    def __init__(self, param_grid, loss_evaluator, output_dir='./sensitivity_results'):
        """Initialize the tester.

        Args:
            param_grid (ParamGrid): Supplies parameter ranges and the baseline
                configuration.
            loss_evaluator (callable): Function that accepts a config dict and
                returns a metrics dict.
            output_dir (str): Directory where result JSON files are written.
        """
        self.param_grid = param_grid
        self.loss_evaluator = loss_evaluator
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Accumulated results from all experiments in this session
        self.results = []

    def grid_search(self, param_names, max_combinations=None, verbose=True):
        """Run a (possibly sub-sampled) factorial grid search.

        Args:
            param_names (list[str]): Dotted parameter names to vary.
            max_combinations (int, optional): Cap on the total number of
                configurations evaluated. When the full grid is larger, random
                sub-sampling is applied.
            verbose (bool): Print per-experiment progress.

        Returns:
            list[dict]: Experiment result dicts, each containing:

                - ``params`` (dict): Parameter values used.
                - ``metrics`` (dict): Metrics returned by the evaluator.
                - ``elapsed`` (float): Wall-clock seconds for this experiment.
                - ``timestamp`` (str): ISO-8601 timestamp.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Grid search over parameters: {param_names}")
            print(f"{'=' * 60}")

        configs = self.param_grid.get_grid_configs(param_names, max_combinations)

        if verbose:
            print(f"Total combinations to test: {len(configs)}")

        results = self._run_experiments(configs, verbose=verbose)

        self.results.extend(results)
        safe_names = '_'.join(p.replace('.', '_') for p in param_names)
        self._save_results(results, f'grid_search_{safe_names}')

        if verbose:
            self._print_top_results(results)

        return results

    def random_search(self, n_configs, param_names=None, verbose=True):
        """Run a random search across the parameter space.

        Args:
            n_configs (int): Number of random configurations to evaluate.
            param_names (list[str], optional): Parameters to randomise.
                Defaults to all parameters in the grid.
            verbose (bool): Print per-experiment progress.

        Returns:
            list[dict]: Experiment result dicts (same schema as
                :meth:`grid_search`).
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Random search ({n_configs} configurations)")
            print(f"{'=' * 60}")

        configs = self.param_grid.get_random_configs(n_configs, param_names)
        results = self._run_experiments(configs, verbose=verbose)

        self.results.extend(results)
        self._save_results(results, 'random_search')

        if verbose:
            self._print_top_results(results)

        return results

    def get_best_config(self, results=None, metric_key='loss', minimize=True):
        """Return the result dict with the best value for ``metric_key``.

        Args:
            results (list[dict], optional): Result list to search. Defaults to
                all accumulated results (``self.results``).
            metric_key (str): Metric to optimise.
            minimize (bool): If ``True``, pick the lowest metric value.

        Returns:
            dict or None: Best result dict, or ``None`` if no results exist.
        """
        if results is None:
            results = self.results
        if not results:
            return None

        valid = [r for r in results if metric_key in r.get('metrics', {})]
        if not valid:
            return None

        metric_values = [r['metrics'][metric_key] for r in valid]
        best_idx = int(np.argmin(metric_values) if minimize else np.argmax(metric_values))
        return valid[best_idx]

    def analyze_sensitivity(self, results=None, metric_key='loss'):
        """Rank each parameter by its influence on ``metric_key``.

        The sensitivity of a parameter is defined as the range (max − min) of
        the mean metric value observed across that parameter's distinct values,
        averaged over all other parameters.

        Args:
            results (list[dict], optional): Experiment results to analyse.
                Defaults to ``self.results``.
            metric_key (str): Metric used for sensitivity estimation.

        Returns:
            dict: Mapping of ``{param_name: {'range': float,
            'mean_by_value': {str_value: float}}}`` sorted in descending
            order of sensitivity.
        """
        if results is None:
            results = self.results
        if not results:
            return {}

        # Collect all parameter names that appear in at least one result
        all_params = set()
        for r in results:
            all_params.update(r.get('params', {}).keys())

        sensitivity = {}
        for param in all_params:
            param_groups = {}
            for r in results:
                value = str(r['params'].get(param))
                metric_val = r.get('metrics', {}).get(metric_key)
                if metric_val is not None:
                    param_groups.setdefault(value, []).append(metric_val)

            if len(param_groups) > 1:
                mean_by_value = {v: float(np.mean(ms)) for v, ms in param_groups.items()}
                metric_range = max(mean_by_value.values()) - min(mean_by_value.values())
                sensitivity[param] = {
                    'range': metric_range,
                    'mean_by_value': mean_by_value,
                }

        # Sort by sensitivity (descending)
        sensitivity = dict(
            sorted(sensitivity.items(), key=lambda x: x[1]['range'], reverse=True)
        )
        return sensitivity

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_experiments(self, configs, verbose=True):
        """Evaluate a list of (config, param_dict) pairs.

        Args:
            configs (list[tuple]): Pairs of ``(config_dict, param_dict)``.
            verbose (bool): Print progress.

        Returns:
            list[dict]: Result dicts for successful experiments.
        """
        results = []
        for i, (config, param_dict) in enumerate(configs):
            start_time = time.time()

            if verbose:
                print(f"\n[{i + 1}/{len(configs)}] Params: {param_dict}")

            try:
                metrics = self.loss_evaluator(config)
                elapsed = time.time() - start_time

                result = {
                    'params': param_dict,
                    'metrics': metrics,
                    'elapsed': elapsed,
                    'timestamp': datetime.now().isoformat(),
                }
                results.append(result)

                if verbose:
                    print(f"  Metrics: {metrics}  ({elapsed:.2f}s)")

            except Exception as exc:
                if verbose:
                    print(f"  FAILED: {exc}")

        return results

    def _save_results(self, results, filename):
        """Persist experiment results to a JSON file.

        Args:
            results (list[dict]): Results to serialise.
            filename (str): Base filename (without ``.json`` extension).

        Returns:
            str: Absolute path of the written file.
        """
        filepath = os.path.join(self.output_dir, f'{filename}.json')
        serialisable = self._make_serializable(results)
        with open(filepath, 'w') as fh:
            json.dump(serialisable, fh, indent=2)
        return filepath

    @staticmethod
    def _make_serializable(obj):
        """Recursively convert numpy scalars / arrays to plain Python types."""
        if isinstance(obj, dict):
            return {k: MultiParamTester._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MultiParamTester._make_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @staticmethod
    def _print_top_results(results, top_n=5, metric_key='loss', minimize=True):
        """Print the top-N configurations ranked by ``metric_key``.

        Args:
            results (list[dict]): Experiment results.
            top_n (int): Number of top results to display.
            metric_key (str): Metric to rank by.
            minimize (bool): If ``True``, lower is better.
        """
        valid = [r for r in results if metric_key in r.get('metrics', {})]
        if not valid:
            return

        sorted_results = sorted(
            valid,
            key=lambda r: r['metrics'][metric_key],
            reverse=not minimize,
        )

        top = sorted_results[:min(top_n, len(sorted_results))]
        print(f"\nTop {len(top)} configurations (by '{metric_key}'):")
        print("-" * 80)
        for rank, result in enumerate(top, 1):
            print(f"  #{rank}: {metric_key} = {result['metrics'][metric_key]:.6f}")
            print(f"       Params: {result['params']}")
