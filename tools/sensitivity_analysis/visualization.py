"""
Visualization tools for ResampleLoss sensitivity analysis.

Provides :class:`SensitivityVisualizer` with methods to produce:

* Per-parameter sensitivity line plots.
* A bar chart comparing sensitivity across all parameters.
* Two-parameter heatmaps.
* An overview grid showing all scanned parameters in a single figure.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


# Use a style that is available in most matplotlib versions
_PREFERRED_STYLES = ['seaborn-v0_8', 'seaborn', 'ggplot', 'default']


def _apply_style():
    for style in _PREFERRED_STYLES:
        if style in plt.style.available:
            plt.style.use(style)
            return
    plt.style.use('default')


class SensitivityVisualizer:
    """Produce publication-quality plots for ResampleLoss sensitivity analysis.

    All ``plot_*`` methods share the same signature convention:

    * ``save=True`` – write the figure to ``output_dir``.
    * ``show=False`` – call ``plt.show()`` (useful in interactive sessions).
    * Return value – absolute path of the saved file, or ``None``.

    Example usage::

        from tools.sensitivity_analysis import SensitivityVisualizer

        viz = SensitivityVisualizer(output_dir='results/plots')
        viz.plot_single_param_scan(scan_result)
        viz.plot_sensitivity_comparison(all_scan_results)
        viz.plot_heatmap(multi_results, ['focal.balance_param', 'focal.gamma'])
    """

    def __init__(self, output_dir='./sensitivity_results/plots'):
        """Initialize the visualizer.

        Args:
            output_dir (str): Directory where plot files are written.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        _apply_style()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_single_param_scan(
        self,
        scan_result,
        metric_keys=None,
        save=True,
        show=False,
    ):
        """Plot metric(s) as a function of a single scanned parameter.

        Args:
            scan_result (dict): Result dict returned by
                :meth:`~single_param_scan.SingleParamScanner.scan`.
            metric_keys (list[str], optional): Which metrics to plot. Defaults
                to all keys present in the first metrics dict.
            save (bool): Save the figure to disk.
            show (bool): Call ``plt.show()`` after rendering.

        Returns:
            str or None: Path of the saved file, or ``None``.
        """
        param_name = scan_result['param_name']
        param_values = scan_result['param_values']
        metrics_list = scan_result['metrics']

        if not metrics_list:
            print(f"No metrics to plot for '{param_name}'")
            return None

        if metric_keys is None:
            metric_keys = list(metrics_list[0].keys())

        n_metrics = len(metric_keys)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(
            f'Sensitivity Analysis: {param_name}',
            fontsize=14,
            fontweight='bold',
        )

        for ax, metric_key in zip(axes, metric_keys):
            metric_values = [m.get(metric_key) for m in metrics_list]
            valid_pairs = [
                (p, v) for p, v in zip(param_values, metric_values) if v is not None
            ]
            if not valid_pairs:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            valid_params, valid_metrics = zip(*valid_pairs)
            x_labels = [str(v) for v in valid_params]
            x_pos = list(range(len(x_labels)))

            ax.plot(x_pos, valid_metrics, 'bo-', markersize=8, linewidth=2)
            ax.fill_between(x_pos, valid_metrics, alpha=0.2)

            best_idx = int(np.argmin(valid_metrics))
            ax.plot(
                x_pos[best_idx],
                valid_metrics[best_idx],
                'r*',
                markersize=15,
                label=f'Best: {valid_params[best_idx]}',
            )

            ax.set_xlabel(param_name, fontsize=12)
            ax.set_ylabel(metric_key, fontsize=12)
            ax.set_title(f'{metric_key} vs {param_name}', fontsize=11)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self._maybe_save(
            fig,
            f"scan_{param_name.replace('.', '_')}.png",
            save,
            show,
        )
        return filepath

    def plot_sensitivity_comparison(
        self,
        all_scan_results,
        metric_key='loss',
        save=True,
        show=False,
    ):
        """Bar chart comparing parameter sensitivity across all parameters.

        Sensitivity is measured as the range (max − min) of the metric values
        observed while scanning each parameter individually.

        Args:
            all_scan_results (dict): Mapping of ``{param_name: scan_result}``.
            metric_key (str): Metric to use for the comparison.
            save (bool): Save the figure to disk.
            show (bool): Call ``plt.show()`` after rendering.

        Returns:
            str or None: Path of the saved file, or ``None``.
        """
        sensitivities = {}
        for param_name, result in all_scan_results.items():
            values = [
                m.get(metric_key)
                for m in result.get('metrics', [])
                if m.get(metric_key) is not None
            ]
            if len(values) > 1:
                sensitivities[param_name] = max(values) - min(values)

        if not sensitivities:
            print("No sensitivity data to plot.")
            return None

        sorted_params = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params]
        sensitivity_values = [p[1] for p in sorted_params]
        max_sensitivity = max(sensitivity_values)

        fig, ax = plt.subplots(figsize=(10, max(4, len(param_names) * 0.7)))
        bars = ax.barh(param_names, sensitivity_values, color='steelblue', alpha=0.8)

        for bar, value in zip(bars, sensitivity_values):
            ax.text(
                bar.get_width() + max_sensitivity * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}',
                va='center',
                fontsize=10,
            )

        ax.set_xlabel(f'Sensitivity – range of {metric_key}', fontsize=12)
        ax.set_title(
            f'Parameter Sensitivity Comparison (metric: {metric_key})',
            fontsize=14,
        )
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        filepath = self._maybe_save(
            fig,
            f'sensitivity_comparison_{metric_key}.png',
            save,
            show,
        )
        return filepath

    def plot_heatmap(
        self,
        multi_param_results,
        param_names,
        metric_key='loss',
        save=True,
        show=False,
    ):
        """Heatmap of a metric for a pair of parameters.

        Args:
            multi_param_results (list[dict]): Results from
                :class:`~multi_param_test.MultiParamTester`.
            param_names (list[str]): Exactly two parameter names.
            metric_key (str): Metric to visualise.
            save (bool): Save the figure to disk.
            show (bool): Call ``plt.show()`` after rendering.

        Returns:
            str or None: Path of the saved file, or ``None``.

        Raises:
            ValueError: If ``param_names`` does not contain exactly two items.
        """
        if len(param_names) != 2:
            raise ValueError("plot_heatmap requires exactly 2 parameter names.")

        p1, p2 = param_names

        p1_values = sorted(
            {str(r['params'][p1]) for r in multi_param_results if p1 in r['params']}
        )
        p2_values = sorted(
            {str(r['params'][p2]) for r in multi_param_results if p2 in r['params']}
        )

        if not p1_values or not p2_values:
            print("Not enough data for heatmap.")
            return None

        matrix = np.full((len(p1_values), len(p2_values)), np.nan)
        for result in multi_param_results:
            if p1 in result['params'] and p2 in result['params']:
                i = p1_values.index(str(result['params'][p1]))
                j = p2_values.index(str(result['params'][p2]))
                val = result.get('metrics', {}).get(metric_key)
                if val is not None:
                    matrix[i, j] = val

        fig, ax = plt.subplots(figsize=(max(8, len(p2_values) * 1.2),
                                        max(6, len(p1_values) * 0.8)))
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label=metric_key)

        ax.set_xticks(range(len(p2_values)))
        ax.set_yticks(range(len(p1_values)))
        ax.set_xticklabels(p2_values, rotation=45, ha='right')
        ax.set_yticklabels(p1_values)
        ax.set_xlabel(p2, fontsize=12)
        ax.set_ylabel(p1, fontsize=12)
        ax.set_title(f'Heatmap: {metric_key} ({p1} vs {p2})', fontsize=14)

        for i in range(len(p1_values)):
            for j in range(len(p2_values)):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j, i, f'{matrix[i, j]:.4f}',
                        ha='center', va='center', fontsize=7,
                    )

        plt.tight_layout()
        safe_p1 = p1.replace('.', '_')
        safe_p2 = p2.replace('.', '_')
        filepath = self._maybe_save(
            fig,
            f'heatmap_{safe_p1}_{safe_p2}_{metric_key}.png',
            save,
            show,
        )
        return filepath

    def plot_all_scans(
        self,
        all_scan_results,
        metric_keys=None,
        save=True,
        show=False,
    ):
        """Overview grid showing all single-parameter scans in one figure.

        Args:
            all_scan_results (dict): Mapping of ``{param_name: scan_result}``.
            metric_keys (list[str], optional): Metrics to include as columns.
                Defaults to all keys in the first result.
            save (bool): Save the figure to disk.
            show (bool): Call ``plt.show()`` after rendering.

        Returns:
            str or None: Path of the saved file, or ``None``.
        """
        n_params = len(all_scan_results)
        if n_params == 0:
            return None

        if metric_keys is None:
            first = next(iter(all_scan_results.values()))
            metric_keys = list(first['metrics'][0].keys()) if first['metrics'] else ['loss']

        n_metrics = len(metric_keys)
        fig, axes = plt.subplots(
            n_params, n_metrics,
            figsize=(6 * n_metrics, 4 * n_params),
            squeeze=False,
        )

        fig.suptitle(
            'Parameter Sensitivity Analysis Overview',
            fontsize=16,
            fontweight='bold',
        )

        for row_idx, (param_name, result) in enumerate(all_scan_results.items()):
            param_values = result['param_values']
            metrics_list = result['metrics']

            for col_idx, metric_key in enumerate(metric_keys):
                ax = axes[row_idx][col_idx]
                metric_values = [m.get(metric_key) for m in metrics_list]
                valid_pairs = [
                    (p, v) for p, v in zip(param_values, metric_values) if v is not None
                ]

                if not valid_pairs:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes)
                    continue

                valid_params, valid_metrics = zip(*valid_pairs)
                x_labels = [str(v) for v in valid_params]
                x_pos = list(range(len(x_labels)))

                ax.plot(x_pos, valid_metrics, 'bo-', markersize=6, linewidth=1.5)
                ax.fill_between(x_pos, valid_metrics, alpha=0.15)

                best_idx = int(np.argmin(valid_metrics))
                ax.plot(x_pos[best_idx], valid_metrics[best_idx], 'r*', markersize=12)

                ax.set_xlabel(param_name, fontsize=9)
                ax.set_ylabel(metric_key, fontsize=9)
                ax.set_title(f'{param_name}', fontsize=10)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self._maybe_save(
            fig, 'all_param_scans_overview.png', save, show
        )
        return filepath

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_save(self, fig, filename, save, show):
        """Optionally save and/or display a figure, then close it.

        Args:
            fig (matplotlib.figure.Figure): The figure to handle.
            filename (str): Filename (basename only) for saving.
            save (bool): Write the figure to ``self.output_dir``.
            show (bool): Call ``plt.show()``.

        Returns:
            str or None: Absolute path of the saved file if ``save=True``.
        """
        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        if show:
            plt.show()
        plt.close(fig)
        return filepath
