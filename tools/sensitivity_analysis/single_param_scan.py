"""
Single parameter scan analysis tool for ResampleLoss sensitivity analysis.

Provides :class:`SingleParamScanner` which iterates over candidate values for
one ResampleLoss parameter at a time, evaluates a user-supplied loss function,
and saves the results to disk.
"""

import copy
import json
import os
import time
from datetime import datetime

import numpy as np


class SingleParamScanner:
    """Scan a single ResampleLoss parameter across its candidate values.

    For each candidate value the scanner:

    1. Constructs a full loss configuration by substituting that value into the
       baseline config supplied by the :class:`~param_grid.ParamGrid`.
    2. Calls ``loss_evaluator(config)`` and records the returned metrics dict.
    3. Persists the per-parameter results as JSON inside ``output_dir``.

    Example usage::

        from tools.sensitivity_analysis import ParamGrid, SingleParamScanner

        grid = ParamGrid()
        scanner = SingleParamScanner(grid, my_evaluator)
        result = scanner.scan('focal.balance_param')
        best_val, best_metrics = scanner.get_best_value(
            'focal.balance_param', metric_key='loss'
        )
    """

    def __init__(self, param_grid, loss_evaluator, output_dir='./sensitivity_results'):
        """Initialize the scanner.

        Args:
            param_grid (ParamGrid): Supplies candidate values and the baseline
                configuration.
            loss_evaluator (callable): Function that accepts a config dict and
                returns a metrics dict (e.g. ``{'loss': 0.42, 'mAP': 0.61}``).
            output_dir (str): Directory where per-parameter JSON result files
                are written.
        """
        self.param_grid = param_grid
        self.loss_evaluator = loss_evaluator
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Accumulated results keyed by parameter name
        self.results = {}

    def scan(self, param_name, verbose=True):
        """Scan a single parameter across all its candidate values.

        Args:
            param_name (str): Dotted parameter name (e.g.
                ``'focal.balance_param'``).
            verbose (bool): Print per-step progress to stdout.

        Returns:
            dict: Result dict with keys:

                - ``param_name`` (str)
                - ``param_values`` (list): Candidate values that were tested.
                - ``metrics`` (list[dict]): Metric dicts returned by the
                  evaluator for each candidate value.
                - ``scan_times`` (list[float]): Wall-clock seconds per step.
                - ``timestamp`` (str): ISO-8601 timestamp of the scan.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Scanning parameter: {param_name}")
            print(f"{'=' * 60}")

        configs = self.param_grid.get_single_param_configs(param_name)

        param_values = []
        metrics_list = []
        scan_times = []

        for i, (config, value) in enumerate(configs):
            start_time = time.time()

            if verbose:
                print(
                    f"[{i + 1}/{len(configs)}] {param_name} = {value!r} ... ",
                    end='',
                    flush=True,
                )

            try:
                metrics = self.loss_evaluator(config)
                elapsed = time.time() - start_time

                param_values.append(value)
                metrics_list.append(metrics)
                scan_times.append(elapsed)

                if verbose:
                    print(f"Done ({elapsed:.2f}s) | Metrics: {metrics}")

            except Exception as exc:
                if verbose:
                    print(f"FAILED: {exc}")

        result = {
            'param_name': param_name,
            'param_values': param_values,
            'metrics': metrics_list,
            'scan_times': scan_times,
            'timestamp': datetime.now().isoformat(),
        }

        self.results[param_name] = result
        self._save_result(param_name, result)

        if verbose:
            self._print_scan_summary(result)

        return result

    def scan_all(self, param_names=None, verbose=True):
        """Scan all parameters sequentially (one at a time).

        Args:
            param_names (list[str], optional): Parameters to scan. Defaults to
                all parameters registered in the :class:`~param_grid.ParamGrid`.
            verbose (bool): Print progress to stdout.

        Returns:
            dict: Mapping of ``{param_name: result_dict}`` for every scanned
                parameter.
        """
        if param_names is None:
            param_names = self.param_grid.get_param_names()

        all_results = {}
        for param_name in param_names:
            result = self.scan(param_name, verbose=verbose)
            all_results[param_name] = result

        return all_results

    def get_best_value(self, param_name, metric_key='loss', minimize=True):
        """Return the parameter value that optimises ``metric_key``.

        Args:
            param_name (str): Parameter name (must have been scanned already).
            metric_key (str): Key inside the metrics dict to optimise.
            minimize (bool): If ``True``, select the value that minimises the
                metric; otherwise maximise.

        Returns:
            tuple: ``(best_value, best_metrics_dict)``

        Raises:
            ValueError: If ``param_name`` has not been scanned yet.
        """
        if param_name not in self.results:
            raise ValueError(
                f"No results for '{param_name}'. Call scan() first."
            )

        result = self.results[param_name]
        values = result['param_values']
        metrics = result['metrics']

        metric_values = [
            m.get(metric_key, float('-inf') if not minimize else float('inf'))
            for m in metrics
        ]

        best_idx = int(np.argmin(metric_values) if minimize else np.argmax(metric_values))
        return values[best_idx], metrics[best_idx]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_result(self, param_name, result):
        """Persist a single-parameter scan result to a JSON file.

        Args:
            param_name (str): Parameter name (used for the filename).
            result (dict): Result dict to serialise.

        Returns:
            str: Absolute path of the written file.
        """
        safe_name = param_name.replace('.', '_')
        filepath = os.path.join(self.output_dir, f'scan_{safe_name}.json')
        serialisable = self._make_serializable(result)
        with open(filepath, 'w') as fh:
            json.dump(serialisable, fh, indent=2)
        return filepath

    @staticmethod
    def _make_serializable(obj):
        """Recursively convert numpy scalars / arrays to plain Python types.

        Args:
            obj: Any Python object.

        Returns:
            A JSON-serialisable version of ``obj``.
        """
        if isinstance(obj, dict):
            return {k: SingleParamScanner._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SingleParamScanner._make_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @staticmethod
    def _print_scan_summary(result):
        """Print a tabular summary of a completed scan.

        Args:
            result (dict): Scan result dict as returned by :meth:`scan`.
        """
        param_name = result['param_name']
        values = result['param_values']
        metrics = result['metrics']

        print(f"\nScan summary for '{param_name}':")
        print(f"  {'Value':<20} {'Metrics'}")
        print(f"  {'-' * 60}")
        for value, metric in zip(values, metrics):
            print(f"  {str(value):<20} {metric}")
        print()
