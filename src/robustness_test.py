#!/usr/bin/env python3
"""
Robustness Test Suite for GRS-MDPI-Finance-2026
Manuscript ID: fintech-4407231

Auto-detects simulation function (run_monte_carlo or run_abm_simulation).
Output: results/robustness_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import inspect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Auto-detect available function
SIM_FUNC = None
SIM_FUNC_NAME = None

try:
    from abm_model import run_monte_carlo
    SIM_FUNC = run_monte_carlo
    SIM_FUNC_NAME = "run_monte_carlo"
except ImportError:
    try:
        from abm_model import run_abm_simulation
        SIM_FUNC = run_abm_simulation
        SIM_FUNC_NAME = "run_abm_simulation"
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

print(f"Detected: {SIM_FUNC_NAME}")
sig = inspect.signature(SIM_FUNC)
func_params = list(sig.parameters.keys())
print(f"Parameters: {func_params}")

os.makedirs("results", exist_ok=True)

# Define sweeps based on detected parameters
SWEEPS = []

if "regulatory_regime" in func_params:
    SWEEPS.append(("regulatory_regime", ["baseline", "voluntary", "mandatory"], "mandatory"))
if "n_runs" in func_params:
    SWEEPS.append(("n_runs", [300, 500, 700], 500))
if "n_steps" in func_params:
    SWEEPS.append(("n_steps", [50, 100, 150], 100))

def extract_metrics(result):
    defaults = {
        "greenium_threshold_bps": np.nan,
        "phase_transition_preserved": False,
        "systemic_loss_pct": np.nan,
        "max_cascade_nodes_pct": np.nan,
        "recovery_quarters": np.nan
    }
    if result is None:
        return defaults
    if isinstance(result, dict):
        return {
            "greenium_threshold_bps": result.get("threshold_bps", result.get("greenium_threshold", np.nan)),
            "phase_transition_preserved": result.get("phase_transition", False),
            "systemic_loss_pct": result.get("systemic_loss", np.nan),
            "max_cascade_nodes_pct": result.get("max_cascade", np.nan),
            "recovery_quarters": result.get("recovery_time", np.nan)
        }
    try:
        return {
            "greenium_threshold_bps": getattr(result, "threshold_bps", np.nan),
            "phase_transition_preserved": getattr(result, "phase_transition", False),
            "systemic_loss_pct": getattr(result, "systemic_loss", np.nan),
            "max_cascade_nodes_pct": getattr(result, "max_cascade", np.nan),
            "recovery_quarters": getattr(result, "recovery_time", np.nan)
        }
    except:
        return defaults

def run_sweep(param_name, values, baseline):
    results = []
    for val in values:
        kwargs = {param_name: val}
        print(f"  Running {param_name} = {val} ...")
        try:
            result = SIM_FUNC(**kwargs)
            metrics = extract_metrics(result)
            results.append({
                "param_name": param_name,
                "param_value": str(val),
                "baseline": str(baseline),
                "status": "success",
                **metrics
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "param_name": param_name,
                "param_value": str(val),
                "baseline": str(baseline),
                "status": f"failed: {e}",
                "greenium_threshold_bps": np.nan,
                "phase_transition_preserved": False,
                "systemic_loss_pct": np.nan,
                "max_cascade_nodes_pct": np.nan,
                "recovery_quarters": np.nan
            })
    return results

def main():
    all_results = []
    print("=" * 60)
    print("GRS Robustness Test Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for param_name, values, baseline in SWEEPS:
        print(f"\n[Sweep] {param_name}")
        all_results.extend(run_sweep(param_name, values, baseline))

    df = pd.DataFrame(all_results)
    output_path = "results/robustness_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Total: {len(df)}, Success: {(df['status'] == 'success').sum()}, Failed: {(df['status'] != 'success').sum()}")

    print("\nDone.")

if __name__ == "__main__":
    main()