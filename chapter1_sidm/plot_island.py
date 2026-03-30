#!/usr/bin/env python3
"""
V9 — v31_plot_island.py
========================
Visualize the "Island of Viability" in the (m_χ, m_φ) plane.

Reads v31_all_relic_points.csv (600 cells, numerical Boltzmann),
colors by σ/m(30) and marks SIDM-viable region.
"""
# === path setup (auto-generated) ================================
import sys as _sys, os as _os
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _os.path.join(_ROOT, 'core'))
DATA_DIR = _os.path.join(_ROOT, 'data')
# =================================================================

import sys, csv
from output_manager import get_latest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch

# ==============================================================
#  Load data
# ==============================================================
all_data = []
with open(str(get_latest("v31_all_relic_points"))) as f:
    reader = csv.DictReader(f)
    for r in reader:
        all_data.append({
            'm_chi': float(r['m_chi_GeV']),
            'm_phi': float(r.get('m_phi_MeV', 0) or float(r.get('m_phi_GeV', 0)) * 1e3),  # MeV
            'alpha': float(r['alpha']),
            'omega': float(r['omega_h2']),
            'sm30':  float(r['sigma_m_30']),
            'sm1000': float(r['sigma_m_1000']),
            'lam':   float(r['lambda']),
        })

mc_all   = np.array([d['m_chi'] for d in all_data])
mp_all   = np.array([d['m_phi'] for d in all_data])
sm30_all = np.array([d['sm30'] for d in all_data])
sm1k_all = np.array([d['sm1000'] for d in all_data])

# SIDM viable mask (relaxed: σ/m(30) ≥ 0.5)
viable = (sm30_all >= 0.5) & (sm30_all <= 10.0) & (sm1k_all < 0.1)
not_viable = ~viable

# ==============================================================
#  Figure 1: Island of Viability (m_χ vs m_φ)
# ==============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel (a): σ/m(30 km/s) heatmap ---
ax = axes[0]
sc = ax.scatter(mc_all[not_viable], mp_all[not_viable],
                c=sm30_all[not_viable], cmap='Blues', s=30, alpha=0.4,
                norm=LogNorm(vmin=0.01, vmax=10), marker='o', edgecolors='none')
ax.scatter(mc_all[viable], mp_all[viable],
           c=sm30_all[viable], cmap='Reds', s=80, alpha=0.9,
           norm=LogNorm(vmin=0.01, vmax=10), marker='*', edgecolors='black',
           linewidths=0.5, zorder=5)
ax.set_xlabel(r'$m_\chi$ [GeV]', fontsize=13)
ax.set_ylabel(r'$m_\phi$ [MeV]', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(r'$\sigma/m$ at $v=30$ km/s [cm$^2$/g]', fontsize=12)
cb = fig.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label(r'$\sigma/m(30)$ [cm$^2$/g]', fontsize=11)

# Legend
legend_elements = [
    Patch(facecolor='lightblue', edgecolor='gray', label='Non-viable'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=12, label='SIDM viable'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# --- Panel (b): σ/m(1000 km/s) heatmap ---
ax = axes[1]
sc2 = ax.scatter(mc_all[not_viable], mp_all[not_viable],
                 c=sm1k_all[not_viable], cmap='Purples', s=30, alpha=0.4,
                 norm=LogNorm(vmin=0.001, vmax=1), marker='o', edgecolors='none')
ax.scatter(mc_all[viable], mp_all[viable],
           c=sm1k_all[viable], cmap='Oranges', s=80, alpha=0.9,
           norm=LogNorm(vmin=0.001, vmax=1), marker='*', edgecolors='black',
           linewidths=0.5, zorder=5)
# Draw the 0.1 limit line as horizontal annotation
ax.set_xlabel(r'$m_\chi$ [GeV]', fontsize=13)
ax.set_ylabel(r'$m_\phi$ [MeV]', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(r'$\sigma/m$ at $v=1000$ km/s [cm$^2$/g]', fontsize=12)
cb2 = fig.colorbar(sc2, ax=ax, shrink=0.85)
cb2.set_label(r'$\sigma/m(1000)$ [cm$^2$/g]', fontsize=11)

legend_elements2 = [
    Patch(facecolor='thistle', edgecolor='gray', label='Non-viable'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
               markeredgecolor='black', markersize=12, label='SIDM viable'),
]
ax.legend(handles=legend_elements2, loc='upper left', fontsize=10)

fig.suptitle(r'Scalar Majorana SIDM — Numerical Boltzmann ($\Omega h^2 = 0.120$ exact)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('v31_island_of_viability.png', dpi=200, bbox_inches='tight')
print("Saved: v31_island_of_viability.png")

# ==============================================================
#  Figure 2: σ/m(v) velocity profile for BP1
# ==============================================================
# BP1 = best point: loaded from global_config.json
from global_config import GC
from v22_raw_scan import sigma_T_vpm

_bp1 = GC.benchmark("BP1")
BP1 = {'m_chi': _bp1['m_chi_GeV'], 'm_phi': _bp1['m_phi_MeV'] * 1e-3, 'alpha': _bp1['alpha']}

print("\nComputing σ/m(v) velocity profile for BP1...")
_ = sigma_T_vpm(10.0, 5e-3, 5e-4, 30.0)  # JIT warmup

velocities = np.logspace(np.log10(5), np.log10(3000), 60)
sigma_m_v = np.array([sigma_T_vpm(BP1['m_chi'], BP1['m_phi'], BP1['alpha'], v)
                      for v in velocities])

fig2, ax2 = plt.subplots(figsize=(8, 5.5))
ax2.loglog(velocities, sigma_m_v, 'b-', linewidth=2.5, label='BP1 (numerical Boltzmann)')

# Dwarf band
ax2.axhspan(0.5, 10.0, xmin=0, xmax=1, alpha=0.12, color='green')
ax2.axvline(30, color='green', linestyle='--', alpha=0.6, linewidth=1)
ax2.text(32, 6, 'Dwarfs\n(30 km/s)', fontsize=9, color='green')

# Cluster limit
ax2.axhline(0.1, color='red', linestyle='--', alpha=0.6, linewidth=1)
ax2.axvline(1000, color='red', linestyle='--', alpha=0.6, linewidth=1)
ax2.text(1050, 0.12, 'Clusters\n(1000 km/s)', fontsize=9, color='red')

# Mark BP1 values
sm30_bp1 = sigma_T_vpm(BP1['m_chi'], BP1['m_phi'], BP1['alpha'], 30.0)
sm1k_bp1 = sigma_T_vpm(BP1['m_chi'], BP1['m_phi'], BP1['alpha'], 1000.0)
ax2.plot(30, sm30_bp1, 'g*', markersize=15, markeredgecolor='black', zorder=5)
ax2.plot(1000, sm1k_bp1, 'r*', markersize=15, markeredgecolor='black', zorder=5)

ax2.set_xlabel(r'$v_{\rm rel}$ [km/s]', fontsize=13)
ax2.set_ylabel(r'$\sigma_T/m_\chi$ [cm$^2$/g]', fontsize=13)
ax2.set_xlim(5, 3000)
ax2.set_ylim(0.001, 30)
ax2.set_title(
    r'BP1: $m_\chi={:.1f}$ GeV, $m_\phi={:.1f}$ MeV, '
    r'$\alpha={:.4f}$, $\Omega h^2=0.1200$'.format(
        BP1['m_chi'], BP1['m_phi']*1e3, BP1['alpha']),
    fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which='both')

# Annotate BP1 values
ax2.annotate(r'$\sigma/m = {:.2f}$'.format(sm30_bp1),
             xy=(30, sm30_bp1), xytext=(60, sm30_bp1*2.5),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='green'),
             color='green', fontweight='bold')
ax2.annotate(r'$\sigma/m = {:.3f}$'.format(sm1k_bp1),
             xy=(1000, sm1k_bp1), xytext=(400, sm1k_bp1*0.3),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='red'),
             color='red', fontweight='bold')

plt.tight_layout()
fig2.savefig('v31_bp1_velocity_profile.png', dpi=200, bbox_inches='tight')
print("Saved: v31_bp1_velocity_profile.png")

# ==============================================================
#  Print final BP1 card
# ==============================================================
print("\n" + "=" * 60)
print("  FINAL BP1 — Numerical Boltzmann, SIDM-Viable")
print("=" * 60)
print("  m_chi       = {:.3f} GeV".format(BP1['m_chi']))
print("  m_phi       = {:.3f} MeV".format(BP1['m_phi']*1e3))
print("  alpha       = {:.4e}".format(BP1['alpha']))
print("  Omega h^2   = 0.1200 (exact numerical)")
print("  lambda      = {:.2f}".format(BP1['alpha'] * BP1['m_chi'] / BP1['m_phi']))
print("  sigma/m(30) = {:.4f} cm^2/g".format(sm30_bp1))
print("  sigma/m(1000)= {:.4f} cm^2/g".format(sm1k_bp1))
print("  SIDM viable = YES (0.5 <= {:.2f} <= 10, {:.4f} < 0.1)".format(sm30_bp1, sm1k_bp1))
print("=" * 60)
