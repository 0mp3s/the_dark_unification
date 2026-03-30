#!/usr/bin/env python3
"""
Generate the Majorana/Dirac ratio close-up figure for the paper.
Shows R(v) = sigma_T^Maj / sigma_T^Dir with inset of dR/d(log v).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import csv, os

# ── Load data ──────────────────────────────────────────────────────
DATA = os.path.join(os.path.dirname(__file__), '..', '..',
                    'Secluded-Majorana-SIDM', 'predictions',
                    'majorana_vs_dirac', 'output', 'maj_vs_dir_data.csv')

v, ratio_BP1, ratio_MAP = [], [], []
with open(DATA) as f:
    reader = csv.DictReader(f)
    for row in reader:
        v.append(float(row['v_km_s']))
        ratio_BP1.append(float(row['ratio_BP1']))
        ratio_MAP.append(float(row['ratio_MAP']))

v = np.array(v)
ratio_BP1 = np.array(ratio_BP1)
ratio_MAP = np.array(ratio_MAP)

# ── Compute derivative d R / d(log v) via finite differences ──────
logv = np.log10(v)
dR_MAP  = np.gradient(ratio_MAP, logv)
dR_BP1  = np.gradient(ratio_BP1, logv)

# ── Figure ────────────────────────────────────────────────────────
fig, (ax_main, ax_deriv) = plt.subplots(2, 1, figsize=(7, 7.5),
                                         gridspec_kw={'height_ratios': [3, 2]},
                                         sharex=True)
fig.subplots_adjust(hspace=0.08)

# --- Top panel: R(v) ---
ax_main.axhline(1.0, color='grey', ls='--', lw=0.8, zorder=0)
ax_main.axhline(0.5, color='grey', ls=':', lw=0.7, zorder=0, alpha=0.5)

ax_main.plot(v, ratio_MAP, '-', color='#4CAF50', lw=2.2, label=r'MAP ($\lambda=48.6$)')
ax_main.plot(v, ratio_BP1, '-', color='#2196F3', lw=2.0, label=r'BP1 ($\lambda=3.2$)')

# Mark the R > 1 zone for MAP
mask = ratio_MAP > 1.0
if mask.any():
    v_above = v[mask]
    ax_main.axvspan(v_above[0], v_above[-1], alpha=0.12, color='#4CAF50',
                    label=r'$R > 1$ zone (MAP)')

# Mark peak
i_peak = np.argmax(ratio_MAP)
ax_main.plot(v[i_peak], ratio_MAP[i_peak], 'o', color='#2E7D32', ms=8, zorder=5)
ax_main.annotate(f'$R_{{\\max}} = {ratio_MAP[i_peak]:.3f}$\n$v = {v[i_peak]:.0f}$ km/s',
                 xy=(v[i_peak], ratio_MAP[i_peak]),
                 xytext=(200, ratio_MAP[i_peak] + 0.03),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5),
                 color='#2E7D32', fontweight='bold')

# Asymptote labels
ax_main.text(6, 0.52, r'Born limit: $R \to 1/2$', fontsize=9, color='grey', style='italic')
ax_main.text(1200, 1.035, r'$R \to 1$', fontsize=9, color='grey', style='italic')

# Velocity scale markers
for vv, label in [(30, 'dSph'), (220, 'MW'), (1200, 'Cluster')]:
    ax_main.axvline(vv, color='#9E9E9E', ls=':', lw=0.7, alpha=0.6)
    ax_main.text(vv, 0.46, label, fontsize=8, ha='center', color='#757575',
                 rotation=90, va='bottom')

ax_main.set_ylabel(r'$R(v) = \sigma_T^{\mathrm{Maj}} / \sigma_T^{\mathrm{Dir}}$', fontsize=13)
ax_main.set_ylim(0.42, 1.15)
ax_main.set_xlim(5, 2000)
ax_main.set_xscale('log')
ax_main.legend(loc='center left', fontsize=10, framealpha=0.9)
ax_main.set_title(r'Majorana vs Dirac fingerprint: identical $\delta_\ell$, different statistics',
                  fontsize=12, pad=8)

# --- Bottom panel: dR / d(log v) ---
ax_deriv.axhline(0.0, color='grey', ls='--', lw=0.8)

ax_deriv.plot(v, dR_MAP, '-', color='#4CAF50', lw=2.0, label='MAP')
ax_deriv.plot(v, dR_BP1, '-', color='#2196F3', lw=1.8, label='BP1')

# Zero crossings of MAP derivative → extrema of R
sign_changes = np.where(np.diff(np.sign(dR_MAP)))[0]
for idx in sign_changes:
    if dR_MAP[idx] > 0:  # max
        ax_deriv.plot(v[idx], 0, 'v', color='#2E7D32', ms=10, zorder=5)
        ax_deriv.annotate(f'$R_{{\\max}}$ at {v[idx]:.0f} km/s',
                         xy=(v[idx], 0), xytext=(v[idx]*3, 0.15),
                         fontsize=9, arrowprops=dict(arrowstyle='->', color='#2E7D32'),
                         color='#2E7D32')
    else:  # min
        ax_deriv.plot(v[idx], 0, '^', color='#E91E63', ms=8, zorder=5)

# Velocity scale markers
for vv, label in [(30, 'dSph'), (220, 'MW'), (1200, 'Cluster')]:
    ax_deriv.axvline(vv, color='#9E9E9E', ls=':', lw=0.7, alpha=0.6)

ax_deriv.set_xlabel(r'$v$ [km/s]', fontsize=13)
ax_deriv.set_ylabel(r'$dR / d(\log v)$', fontsize=13)
ax_deriv.legend(loc='upper right', fontsize=10)
ax_deriv.set_ylim(-0.35, 0.55)

for ax in (ax_main, ax_deriv):
    ax.tick_params(labelsize=11)

OUT = os.path.join(os.path.dirname(__file__), 'figures', 'fig_majorana_dirac.png')
fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved → {OUT}")
print(f"MAP peak R = {ratio_MAP[i_peak]:.4f} at v = {v[i_peak]:.1f} km/s")
print(f"R>1 zone: {v[mask][0]:.0f} – {v[mask][-1]:.0f} km/s")
plt.close()
