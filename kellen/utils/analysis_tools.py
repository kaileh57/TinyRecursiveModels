"""
Analysis tools for TRM scaling experiments
Parse logs, compute metrics, generate plots
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def parse_wandb_export(export_dir: Path) -> pd.DataFrame:
    """Parse exported WandB runs into DataFrame"""
    # Placeholder - would parse WandB CSV exports
    pass


def compute_scaling_law(sizes: np.ndarray, accuracies: np.ndarray) -> Dict:
    """
    Fit scaling law: accuracy = a - b * size^(-c)
    Returns fitted parameters and predictions
    """
    from scipy.optimize import curve_fit

    def power_law(x, a, b, c):
        return a - b * np.power(x, -c)

    # Fit curve
    try:
        params, _ = curve_fit(power_law, sizes, accuracies, p0=[0.9, 0.1, 0.3])
        predictions = power_law(sizes, *params)

        return {
            "a": params[0],
            "b": params[1],
            "c": params[2],
            "predictions": predictions,
            "formula": f"{params[0]:.3f} - {params[1]:.3f} * x^(-{params[2]:.3f})"
        }
    except Exception as e:
        print(f"Warning: Scaling law fit failed: {e}")
        return None


def plot_model_size_scaling(results_df: pd.DataFrame, output_path: Path):
    """Plot accuracy vs model size with scaling law fit"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data
    sizes = results_df['params'].values
    accuracies = results_df['test_accuracy'].values

    # Plot 1: Accuracy vs Parameters
    ax1.scatter(sizes, accuracies, s=100, alpha=0.7, label='Experiments')

    # Fit scaling law
    scaling_law = compute_scaling_law(sizes, accuracies)
    if scaling_law:
        ax1.plot(sizes, scaling_law['predictions'], 'r--', label=f"Fit: {scaling_law['formula']}")

    ax1.set_xlabel('Parameters (M)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Model Size Scaling', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Efficiency (accuracy / params)
    efficiency = accuracies / (sizes / 1e6)  # accuracy per million params
    ax2.scatter(sizes, efficiency, s=100, alpha=0.7, color='green')
    ax2.set_xlabel('Parameters (M)', fontsize=12)
    ax2.set_ylabel('Efficiency (Acc / M params)', fontsize=12)
    ax2.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")


def plot_depth_scaling(results_df: pd.DataFrame, output_path: Path):
    """Plot accuracy vs recursion depth"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate L_cycles and H_cycles experiments
    l_cycles_data = results_df[results_df['experiment'].str.contains('exp02a')]
    h_cycles_data = results_df[results_df['experiment'].str.contains('exp02b')]

    if not l_cycles_data.empty:
        ax.plot(l_cycles_data['L_cycles'], l_cycles_data['test_accuracy'],
                'o-', label='L_cycles (latent updates)', linewidth=2, markersize=8)

    if not h_cycles_data.empty:
        ax.plot(h_cycles_data['H_cycles'], h_cycles_data['test_accuracy'],
                's-', label='H_cycles (high-level reasoning)', linewidth=2, markersize=8)

    ax.set_xlabel('Cycle Count', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Recursion Depth Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")


def plot_batch_size_scaling(results_df: pd.DataFrame, output_path: Path):
    """Plot throughput and accuracy vs batch size"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    batch_sizes = results_df['global_batch_size'].values
    accuracies = results_df['test_accuracy'].values
    throughputs = results_df['throughput'].values  # examples/sec

    # Plot 1: Accuracy vs Batch Size
    ax1.plot(batch_sizes, accuracies, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Global Batch Size', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Batch Size vs Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Throughput vs Batch Size
    ax2.plot(batch_sizes, throughputs, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Global Batch Size', fontsize=12)
    ax2.set_ylabel('Throughput (examples/sec)', fontsize=12)
    ax2.set_title('Batch Size vs Throughput', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")


def generate_experiment_report(experiment_group: str, results_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive report for an experiment group"""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append(f"# TRM Scaling Experiment Report: {experiment_group}")
    report.append(f"\nGenerated: {pd.Timestamp.now()}\n")
    report.append("="*80)

    # Summary statistics
    report.append("\n## Summary Statistics\n")
    report.append(f"- Total experiments: {len(results_df)}")
    report.append(f"- Mean accuracy: {results_df['test_accuracy'].mean():.2f}%")
    report.append(f"- Best accuracy: {results_df['test_accuracy'].max():.2f}%")
    report.append(f"- Worst accuracy: {results_df['test_accuracy'].min():.2f}%")

    # Best configuration
    best_idx = results_df['test_accuracy'].idxmax()
    best_config = results_df.loc[best_idx]
    report.append(f"\n## Best Configuration\n")
    report.append(f"- Experiment: {best_config['experiment']}")
    report.append(f"- Accuracy: {best_config['test_accuracy']:.2f}%")
    report.append(f"- Parameters: {best_config.get('params', 'N/A')}")

    # Detailed results table
    report.append(f"\n## Detailed Results\n")
    report.append(results_df.to_markdown(index=False))

    # Save report
    report_path = output_dir / f"{experiment_group}_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report))

    print(f"Saved report: {report_path}")


def main():
    """Example usage"""
    print("Analysis tools loaded. Import and use functions as needed.")


if __name__ == "__main__":
    main()
