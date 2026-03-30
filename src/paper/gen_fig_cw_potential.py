"""Generate Coleman-Weinberg potential V_CW(theta) figure for Chapter 2."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CW potential for Majorana fermion with A4 symmetry
# V_CW(theta) = -2/(64 pi^2) * M_eff^4(theta) * [ln(M_eff^2/mu^2) - 3/2]
# M_eff(theta) = m_chi * sqrt(cos^2(theta) + sin^2(theta)/9)

theta = np.linspace(0, np.pi, 500)
m_chi = 94.07  # MAP benchmark, GeV
mu = m_chi  # renormalization scale

def M_eff(th):
    return m_chi * np.sqrt(np.cos(th)**2 + np.sin(th)**2 / 9)

def V_CW(th):
    M = M_eff(th)
    return -(2 / (64 * np.pi**2)) * M**4 * (np.log(M**2 / mu**2) - 1.5)

V = V_CW(theta)
V_norm = (V - V.min()) / (V.max() - V.min())  # normalize to [0,1]

# Find minimum
i_min = np.argmin(V)
theta_min = theta[i_min]

# A4 discrete value: theta = arcsin(1/3) ≈ 19.47 degrees ≈ 0.3398 rad
theta_A4 = np.arcsin(1/3)
# The relic angle
theta_relic_deg = np.degrees(np.arctan(1/np.sqrt(8)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Full CW potential
ax1.plot(np.degrees(theta), V_norm, 'b-', linewidth=2)
ax1.axvline(np.degrees(theta_min), color='r', linestyle='--', alpha=0.7,
            label=f'CW minimum: $\\theta = {np.degrees(theta_min):.1f}°$')
ax1.axvline(np.degrees(theta_A4), color='g', linestyle=':', alpha=0.7, linewidth=2,
            label=f'$A_4$ value: $\\theta = {np.degrees(theta_A4):.1f}°$')
ax1.axvline(theta_relic_deg, color='purple', linestyle='-.', alpha=0.7, linewidth=2,
            label=f'$\\theta_{{relic}} = {theta_relic_deg:.1f}°$')
ax1.set_xlabel('CP angle $\\theta$ [degrees]', fontsize=13)
ax1.set_ylabel('$V_{CW}(\\theta)$ [normalized]', fontsize=13)
ax1.set_title('Coleman-Weinberg Potential', fontsize=14)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_xlim(0, 180)
ax1.grid(True, alpha=0.3)

# Panel 2: Zoom near the minimum with theta_relic marked
theta_zoom = np.linspace(0, 60, 500)
theta_zoom_rad = np.radians(theta_zoom)
V_zoom = V_CW(theta_zoom_rad)
V_zoom_norm = (V_zoom - V.min()) / (V.max() - V.min())

ax2.plot(theta_zoom, V_zoom_norm, 'b-', linewidth=2)
ax2.axvline(theta_relic_deg, color='purple', linestyle='-.', linewidth=2,
            label=f'$\\theta_{{relic}} = {theta_relic_deg:.2f}°$')
ax2.axvline(np.degrees(theta_A4), color='g', linestyle=':', linewidth=2,
            label=f'$A_4$: $\\theta = {np.degrees(theta_A4):.2f}°$')

# Mark the relic angle point
V_at_relic = V_CW(np.radians(theta_relic_deg))
V_at_relic_norm = (V_at_relic - V.min()) / (V.max() - V.min())
ax2.plot(theta_relic_deg, V_at_relic_norm, 'o', color='purple', markersize=10, zorder=5)

# Shade the DM and DE regions
ax2.axvspan(0, theta_relic_deg, alpha=0.1, color='green', label='DM-dominated')
ax2.axvspan(theta_relic_deg, 60, alpha=0.1, color='purple', label='DE contribution')

ax2.set_xlabel('CP angle $\\theta$ [degrees]', fontsize=13)
ax2.set_ylabel('$V_{CW}(\\theta)$ [normalized]', fontsize=13)
ax2.set_title('Zoom: Relic Angle Region', fontsize=14)
ax2.legend(fontsize=9, loc='upper right')
ax2.set_xlim(0, 60)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = Path('figures/fig_cw_potential.png')
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")
print(f"CW minimum at theta = {np.degrees(theta_min):.1f}°")
print(f"A4 value: theta = {np.degrees(theta_A4):.2f}°")
print(f"Relic angle: theta = {theta_relic_deg:.2f}°")
