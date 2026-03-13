"""
Experiment report generation tool for ResampleLoss sensitivity analysis.

Provides :class:`ReportGenerator` which creates human-readable text reports
and machine-readable JSON summaries from sensitivity-analysis result objects.
"""

import json
import os
from datetime import datetime

import numpy as np


class ReportGenerator:
    """Generate text and JSON reports from sensitivity analysis experiments.

    Example usage::

        from tools.sensitivity_analysis import ReportGenerator

        reporter = ReportGenerator(output_dir='sensitivity_results')
        reporter.generate_scan_report(scan_results, metric_key='loss')
        reporter.generate_multi_param_report(multi_results, metric_key='loss')
        reporter.generate_json_summary(scan_results, multi_results)
    """

    def __init__(self, output_dir='./sensitivity_results'):
        """Initialize the report generator.

        Args:
            output_dir (str): Directory where report files are written.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_scan_report(
        self,
        scan_results,
        metric_key='loss',
        output_file=None,
    ):
        """Generate a detailed text report for single-parameter scan results.

        The report includes:

        * An executive summary ranking parameters by their measured sensitivity.
        * A detailed per-parameter section with a tabular breakdown of metric
          values and the best parameter value found.

        Args:
            scan_results (dict): Mapping of ``{param_name: result_dict}``
                as returned by :meth:`~single_param_scan.SingleParamScanner.scan_all`.
            metric_key (str): Primary metric to highlight.
            output_file (str, optional): Destination path. Auto-generated from
                the current timestamp when not provided.

        Returns:
            str: Absolute path of the written report file.
        """
        if output_file is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f'scan_report_{ts}.txt')

        lines = []
        self._header(lines, 'RESAMPLELOSS PARAMETER SENSITIVITY ANALYSIS REPORT')
        lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Primary metric : {metric_key}")
        lines.append('')

        # ---- Executive summary ----------------------------------------
        lines.append('EXECUTIVE SUMMARY')
        lines.append('-' * 40)
        sensitivities = self._compute_scan_sensitivities(scan_results, metric_key)
        ranked = sorted(sensitivities.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
        lines.append('Parameter sensitivity ranking (highest → lowest range of metric):')
        for rank, (param, stats) in enumerate(ranked, 1):
            line = (
                '  #{:>2}  {:<35} range={:.6f}'
                '  (min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f})'
            ).format(
                rank, param, stats['sensitivity'],
                stats['min'], stats['max'], stats['mean'], stats['std'],
            )
            lines.append(line)
        lines.append('')

        # ---- Per-parameter details ------------------------------------
        lines.append('DETAILED PARAMETER ANALYSIS')
        lines.append('-' * 40)
        for param_name, result in scan_results.items():
            self._append_param_section(lines, param_name, result, metric_key)

        self._footer(lines)
        return self._write(lines, output_file)

    def generate_multi_param_report(
        self,
        multi_param_results,
        metric_key='loss',
        output_file=None,
    ):
        """Generate a text report for multi-parameter experiment results.

        The report includes:

        * Top-10 configurations ranked by the chosen metric.
        * Per-parameter sensitivity analysis averaged over all experiments.

        Args:
            multi_param_results (list[dict]): Results from
                :class:`~multi_param_test.MultiParamTester`.
            metric_key (str): Primary metric to highlight.
            output_file (str, optional): Destination path. Auto-generated when
                not provided.

        Returns:
            str: Absolute path of the written report file.
        """
        if output_file is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f'multi_param_report_{ts}.txt')

        lines = []
        self._header(lines, 'MULTI-PARAMETER SENSITIVITY ANALYSIS REPORT')
        lines.append(f"Generated          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total experiments  : {len(multi_param_results)}")
        lines.append(f"Primary metric     : {metric_key}")
        lines.append('')

        valid = [r for r in multi_param_results if metric_key in r.get('metrics', {})]
        if not valid:
            lines.append('No valid results found.')
            self._footer(lines)
            return self._write(lines, output_file)

        sorted_results = sorted(valid, key=lambda r: r['metrics'][metric_key])

        # ---- Top configurations ----------------------------------------
        lines.append('TOP 10 CONFIGURATIONS')
        lines.append('-' * 40)
        for i, result in enumerate(sorted_results[:10], 1):
            lines.append(f'\n  #{i}  {metric_key} = {result["metrics"][metric_key]:.6f}')
            for pname, pvalue in result['params'].items():
                lines.append(f'       {pname}: {pvalue}')

        lines.append('')

        # ---- Per-parameter sensitivity ---------------------------------
        lines.append('PARAMETER SENSITIVITY ANALYSIS')
        lines.append('-' * 40)
        all_params = sorted({p for r in valid for p in r['params']})
        for param in all_params:
            param_groups = {}
            for r in valid:
                if param in r['params']:
                    key = str(r['params'][param])
                    param_groups.setdefault(key, []).append(r['metrics'][metric_key])

            if len(param_groups) < 2:
                continue

            lines.append(f'\n  Parameter: {param}')
            mean_by_value = {v: float(np.mean(ms)) for v, ms in param_groups.items()}
            for value, mean in sorted(mean_by_value.items(), key=lambda x: x[1]):
                n = len(param_groups[value])
                lines.append(f'    {value:<25} mean={mean:.6f}  (n={n})')

            sensitivity = max(mean_by_value.values()) - min(mean_by_value.values())
            lines.append(f'    Sensitivity (range of means): {sensitivity:.6f}')

        lines.append('')
        self._footer(lines)
        return self._write(lines, output_file)

    def generate_json_summary(
        self,
        scan_results,
        multi_param_results=None,
        output_file=None,
    ):
        """Write a JSON summary of all results for programmatic consumption.

        Args:
            scan_results (dict): Single-parameter scan results.
            multi_param_results (list[dict], optional): Multi-parameter results.
            output_file (str, optional): Destination path. Auto-generated when
                not provided.

        Returns:
            str: Absolute path of the written JSON file.
        """
        if output_file is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f'summary_{ts}.json')

        summary = {
            'timestamp': datetime.now().isoformat(),
            'single_param_scans': scan_results,
            'multi_param_results': multi_param_results or [],
        }

        summary = self._make_serializable(summary)
        with open(output_file, 'w') as fh:
            json.dump(summary, fh, indent=2)

        print(f"JSON summary saved to: {output_file}")
        return output_file

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_scan_sensitivities(self, scan_results, metric_key):
        """Compute sensitivity statistics for each scanned parameter.

        Args:
            scan_results (dict): Scan result mapping.
            metric_key (str): Metric to analyse.

        Returns:
            dict: ``{param_name: {'sensitivity', 'min', 'max', 'mean', 'std'}}``
        """
        sensitivities = {}
        for param_name, result in scan_results.items():
            values = [
                m.get(metric_key)
                for m in result.get('metrics', [])
                if m.get(metric_key) is not None
            ]
            if len(values) > 1:
                sensitivities[param_name] = {
                    'sensitivity': max(values) - min(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }
        return sensitivities

    def _append_param_section(self, lines, param_name, result, metric_key):
        """Append a detailed section for one parameter to ``lines``.

        Args:
            lines (list[str]): Accumulator for report lines.
            param_name (str): Parameter name.
            result (dict): Scan result dict.
            metric_key (str): Primary metric.
        """
        lines.append(f'\n  Parameter: {param_name}')
        lines.append('  ' + '~' * 50)

        param_values = result.get('param_values', [])
        metrics_list = result.get('metrics', [])

        if not param_values or not metrics_list:
            lines.append('    No data available.')
            return

        all_metric_keys = list(metrics_list[0].keys()) if metrics_list else []
        col_w = 15
        header = f"    {'Value':<20} " + ' '.join(f'{k:<{col_w}}' for k in all_metric_keys)
        lines.append(header)
        lines.append('    ' + '-' * (20 + (col_w + 1) * len(all_metric_keys)))

        # Track best value per metric
        best_by_metric = {k: (None, float('inf')) for k in all_metric_keys}

        for value, metrics in zip(param_values, metrics_list):
            row = f"    {str(value):<20} "
            for k in all_metric_keys:
                v = metrics.get(k)
                if v is not None:
                    row += f'{v:<{col_w}.6f} '
                    if v < best_by_metric[k][1]:
                        best_by_metric[k] = (value, v)
                else:
                    row += f'{"N/A":<{col_w}} '
            lines.append(row)

        lines.append(f"\n    Best values (lowest {metric_key}):")
        for k, (best_val, best_metric) in best_by_metric.items():
            if best_val is not None:
                lines.append(f'      {k}: best param value = {best_val}  '
                             f'(metric = {best_metric:.6f})')

    @staticmethod
    def _header(lines, title):
        lines.append('=' * 80)
        lines.append(title)
        lines.append('=' * 80)

    @staticmethod
    def _footer(lines):
        lines.append('=' * 80)
        lines.append('END OF REPORT')
        lines.append('=' * 80)

    def _write(self, lines, output_file):
        """Write ``lines`` to ``output_file`` and return the path."""
        text = '\n'.join(lines)
        with open(output_file, 'w') as fh:
            fh.write(text)
        print(f"Report saved to: {output_file}")
        return output_file

    @staticmethod
    def _make_serializable(obj):
        """Recursively convert numpy scalars / arrays to plain Python types."""
        if isinstance(obj, dict):
            return {k: ReportGenerator._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ReportGenerator._make_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
