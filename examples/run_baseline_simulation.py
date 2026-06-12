"""
Example: Run Baseline GRS Simulation
Author: Chengcheng Gan
"""

import sys
sys.path.insert(0, 'src')

from src.abm_model import ESGMarketModel, run_monte_carlo
from src.network_builder import TemporalNetwork
from src.regulatory_sandbox import RegulatorySandbox
import matplotlib.pyplot as plt
import numpy as np


def run_baseline_example():
    """Run single simulation example."""
    print("=" * 60)
    print("Greenium Resilience Simulator - Baseline Example")
    print("=" * 60)

    # 1. Initialize model
    print("
[1] Initializing ESG Market Model...")
    model = ESGMarketModel(
        n_agents=100,
        regulatory_regime='mandatory',
        network_density=0.15,
        seed=42
    )
    print(f"    Agents: {model.n_agents}")
    print(f"    Regime: {model.regulatory_regime}")
    print(f"    Network density: {model.network_density}")

    # 2. Run simulation
    print("
[2] Running simulation (100 steps)...")
    results = model.run_simulation(n_steps=100)

    # 3. Display results
    print("
[3] Simulation Results:")
    print(f"    Final greenium: {results['Greenium'].iloc[-1]:.2f} bps")
    print(f"    Average greenium: {results['Greenium'].mean():.2f} bps")
    print(f"    Greenium volatility: {results['Greenium'].std():.2f} bps")
    print(f"    Final avg belief (GW): {results['Avg_Belief_GW'].iloc[-1]:.3f}")
    print(f"    Number of climate shocks: {len(model.shock_history)}")
    print(f"    Max network centrality: {results['Network_Centrality_Max'].iloc[-1]:.4f}")

    # 4. Save results
    print("
[4] Saving results...")
    results.to_csv('outputs/simulation_results/baseline_run.csv', index=False)
    print("    Saved: outputs/simulation_results/baseline_run.csv")

    # 5. Generate plot
    print("
[5] Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Greenium dynamics
    axes[0, 0].plot(results['Greenium'], color='green', linewidth=2)
    axes[0, 0].axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Greenium (bps)')
    axes[0, 0].set_title('Greenium Dynamics Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Belief evolution
    axes[0, 1].plot(results['Avg_Belief_GW'], color='red', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Average Belief in GW')
    axes[0, 1].set_title('Investor Belief in Greenwashing')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Shock timeline
    shock_steps = [s['step'] for s in model.shock_history]
    shock_severities = [s['severity'] for s in model.shock_history]
    if shock_steps:
        axes[1, 0].scatter(shock_steps, shock_severities, color='orange', s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Shock Severity')
    axes[1, 0].set_title('Climate Shock Events')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Network centrality
    axes[1, 1].plot(results['Network_Centrality_Max'], color='blue', linewidth=2)
    axes[1, 1].axhline(y=0.23, color='red', linestyle='--', alpha=0.5, label='Threshold (0.23)')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Max Eigenvector Centrality')
    axes[1, 1].set_title('Network Centrality Dynamics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/baseline_simulation.png', dpi=300, bbox_inches='tight')
    print("    Saved: outputs/figures/baseline_simulation.png")

    plt.close()
    print("
" + "=" * 60)
    print("Baseline example complete!")
    print("=" * 60)


def run_monte_carlo_example():
    """Run Monte Carlo comparison across regimes."""
    print("
" + "=" * 60)
    print("Monte Carlo Comparison: Mandatory vs. Voluntary")
    print("=" * 60)

    regimes = ['mandatory', 'voluntary']
    n_runs = 50  # Reduced for quick demo

    all_results = {}

    for regime in regimes:
        print(f"
Running {regime} regime ({n_runs} runs)...")
        mc_results = run_monte_carlo(n_runs=n_runs, n_steps=100, regulatory_regime=regime)
        all_results[regime] = mc_results

        print(f"  Mean final greenium: {mc_results['final_greenium'].mean():.2f} bps")
        print(f"  Std final greenium: {mc_results['final_greenium'].std():.2f} bps")
        print(f"  Mean volatility: {mc_results['volatility'].mean():.2f} bps")

    # Compare
    mandatory_mean = all_results['mandatory']['final_greenium'].mean()
    voluntary_mean = all_results['voluntary']['final_greenium'].mean()
    attenuation = (voluntary_mean - mandatory_mean) / voluntary_mean * 100

    print(f"
Comparison:")
    print(f"  Mandatory attenuation: {attenuation:.1f}%")
    print(f"  (Positive = mandatory preserves more greenium)")

    # Save comparison
    comparison = pd.DataFrame({
        'regime': ['mandatory', 'voluntary'],
        'mean_greenium': [mandatory_mean, voluntary_mean],
        'std_greenium': [all_results['mandatory']['final_greenium'].std(),
                        all_results['voluntary']['final_greenium'].std()]
    })
    comparison.to_csv('outputs/simulation_results/regime_comparison.csv', index=False)
    print("
  Saved: outputs/simulation_results/regime_comparison.csv")


if __name__ == '__main__':
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/simulation_results', exist_ok=True)

    # Run examples
    run_baseline_example()
    run_monte_carlo_example()

    print("
" + "=" * 60)
    print("All examples complete. Check outputs/ directory.")
    print("=" * 60)
