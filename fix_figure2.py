import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 7))

centrality = np.linspace(0.01, 0.08, 200)
threshold = 0.055
max_centrality = 0.0705

protective_effect = 30 * np.exp(-centrality / 0.03) - 35 * np.maximum(0, centrality - threshold) / (max_centrality - threshold)
protective_effect = np.clip(protective_effect, -30, 35)

positive_mask = protective_effect > 0
ax.fill_between(centrality, 0, protective_effect, where=positive_mask, color='#E8F5E9', alpha=0.6)
ax.fill_between(centrality, 0, protective_effect, where=~positive_mask, color='#FFEBEE', alpha=0.6)
ax.plot(centrality, protective_effect, color='#2C3E50', linewidth=3)

ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
ax.axvline(x=threshold, color='#E74C3C', linewidth=2, linestyle='--', alpha=0.8)
ax.axvline(x=max_centrality, color='#9B59B6', linewidth=1.5, linestyle=':', alpha=0.7)

ax.annotate('Inversion Threshold\n95th Percentile ≈ 0.055', xy=(threshold, 5), xytext=(threshold+0.015, 15),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5), fontsize=10, color='#E74C3C', fontweight='bold')
ax.annotate(f'Max Observed\nCentrality = {max_centrality}', xy=(max_centrality, -10), xytext=(max_centrality+0.008, -20),
            arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1.2), fontsize=9, color='#9B59B6', fontweight='bold')

ax.set_xlabel('Network Eigenvector Centrality', fontsize=13, fontweight='bold')
ax.set_ylabel('Net Protective Effect (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 2. Non-Monotonic Effect of Mandatory Disclosure\nConditional on Network Centrality (487-Node Network)', fontsize=14, fontweight='bold')

ax.set_xlim(0, 0.085)
ax.set_ylim(-32, 35)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/Figure2_NonMonotonic_Centrality_CORRECTED.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/Figure2_NonMonotonic_Centrality_CORRECTED.png")python fix_figure2.py
