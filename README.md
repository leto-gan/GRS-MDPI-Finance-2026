# Greenium Resilience Simulator (GRS)

**Agent-Based Network Model for Systemic Repricing Risk in ESG Markets under Climate Shock Scenarios**

This repository contains the Python implementation of the Greenium Resilience Simulator (GRS), a computational framework for modeling how climate-induced repricing shocks propagate through ESG-rated asset networks under divergent regulatory regimes.

## Overview

The GRS integrates three components:
1. **Contextualized Agent-Based Model (ABM)**: Institutional investors with heterogeneous beliefs and Bayesian learning about greenwashing
2. **Multi-Jurisdictional Temporal Network**: Cross-border contagion through shared ESG rating providers, supply chains, and ownership links
3. **Comparative Regulatory Sandbox**: Ex-ante testing of policy interventions (EU SFDR vs. US voluntary disclosure)

## Publication

This code supports the manuscript submitted to **MDPI FinTech**:
> "Systemic Repricing Risk in ESG Markets: An Agent-Based Network Model of Greenium Fragility under Climate Shock Scenarios"

## Installation

```bash
git clone https://github.com/leto-gan/GRS-MDPI-Finance-2026.git
cd GRS-MDPI-Finance-2026
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- NumPy 1.24+
- Pandas 2.0+
- NetworkX 3.1+
- Mesa 2.1+
- Matplotlib 3.7+
- Seaborn 0.12+

## Quick Start

Run the baseline simulation:

```bash
python examples/run_baseline_simulation.py
```

This executes the full GRS pipeline:
1. Loads sample ESG and network data
2. Initializes agents with heterogeneous beliefs
3. Simulates climate shock propagation
4. Outputs network contagion metrics and greenium dynamics

## Data Sources

All data used in this study are publicly available:

- **ESG Ratings**: Yahoo Finance (via `yfinance` library) — free, covers S&P 500 firms
- **Climate Shocks**: EM-DAT International Disaster Database (https://www.emdat.be/) — free academic registration
- **Regulatory Parameters**: EU SFDR (EUR-Lex), SEC Climate Disclosure Rules (sec.gov)
- **Network Data**: SEC 13F filings (public), GICS sector classifications (MSCI, free)

No proprietary data purchases required.

## Repository Structure

```
GRS-MDPI-Finance-2026/
├── src/              # Source code
├── data/             # Sample data and data acquisition scripts
├── examples/         # Runnable examples
├── tests/            # Unit tests
└── outputs/          # Simulation outputs and figures
```

## Usage

### Basic Simulation

```python
from src.abm_model import ESGMarketModel
from src.network_builder import TemporalNetwork
from src.regulatory_sandbox import RegulatorySandbox

# Initialize model
model = ESGMarketModel(
    n_agents=100,
    regulatory_regime='mandatory',  # or 'voluntary'
    network_density=0.15
)

# Run simulation
for step in range(100):
    model.step()

# Extract results
results = model.datacollector.get_model_vars_dataframe()
```

### Network Analysis

```python
from src.network_builder import TemporalNetwork

network = TemporalNetwork()
network.build_from_ownership_data('data/sec_13f_sample.csv')
centrality = network.compute_eigenvector_centrality()
contagion = network.simulate_shock_propagation(seed_nodes=['AAPL', 'MSFT'])
```

## Reproducibility

To reproduce the main results reported in the manuscript:

```bash
python examples/reproduce_main_results.py
```

This runs 1,000 Monte Carlo simulations across four regulatory scenarios and outputs:
- `outputs/figures/figure1_greenium_dynamics.pdf`
- `outputs/figures/figure2_network_contagion.pdf`
- `outputs/figures/figure3_regulatory_comparison.pdf`
- `outputs/tables/table1_simulation_summary.csv`

## Citation

If you use this code, please cite:

```
Gan, C. (2026). Systemic Repricing Risk in ESG Markets: An Agent-Based Network Model 
of Greenium Fragility under Climate Shock Scenarios. MDPI Finance, under review.
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file.

## Contact

Chengcheng Gan — Leto_Gan@163.com
