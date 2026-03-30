#!/usr/bin/env python3
"""
vpm_scan/velocity_slope.py
==========================
Logarithmic velocity slope η(v) = d ln(σ/m) / d ln v  for BP1 and MAP.

Physics:
  η(v) characterises how rapidly σ/m changes with velocity.

  Key regimes:
    η ≈ 0    flat region  — prediction robust to velocity uncertainties
    η ≈ -2   perturbative tail (σ ∝ 1/v²)
    |η| >> 1 near resonance nodes — observation very sensitive to v

  Practical consequence:
    Observations in the flat region (η ≈ 0) are the most reliable
    constraints on σ/m. In the steep region (|η| large) the measurement
    is fragile: a ±20% velocity uncertainty shifts σ/m by ±20|η| %.

Produces:
  - Console summary of key regimes
  - output/velocity_slope.png  (3-panel: σ/m, η, robustness map)
  - output/velocity_slope.pdf
  - output/velocity_slope.csv
"""
import sys, os, math, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, '..')
sys.path.insert(0, os.path.join(_ROOT, 'core'))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config_loader import load_config
from global_config import GC
from v22_raw_scan import sigma_T_vpm

# Warm up JIT
sigma_T_vpm(20.0, 10e-3, 1e-3, 100.0)

cfg = load_config(__file__)
_bps = {bp['label']: bp for bp in GC.benchmarks_from_labels(cfg.get('benchmark_labels', ['BP1', 'MAP']))}
BP1 = _bps['BP1']
MAP = _bps['MAP']

# ── velocity grid (log-spaced) ──
N_PTS = 500
V_MIN, V_MAX = 3.0, 5000.0
V_GRID = np.logspace(np.log10(V_MIN), np.log10(V_MAX), N_PTS)

# Astrophysical reference velocities
V_REFS = {'dSph': 30, 'MW': 220, 'Cluster': 1200}


def compute_sigma_grid(bp):
    """Compute σ_T/m [cm²/g] on the velocity grid."""
    m_chi = bp['m_chi_GeV']
    m_phi = bp['m_phi_MeV'] / 1000.0
    alpha = bp['alpha']
    return np.array([sigma_T_vpm(m_chi, m_phi, alpha, v) for v in V_GRID])


def log_slope(log_v, log_sigma):
    """Compute η = d(ln σ)/d(ln v) via central differences on log-spaced grid."""
    eta = np.gradient(log_sigma, log_v)
    return eta


def find_flat_regions(v, eta, threshold=0.3):
    """Return velocity intervals where |η| < threshold (robust region)."""
    flat = np.abs(eta) < threshold
    regions = []
    in_region = False
    for i, is_flat in enumerate(flat):
        if is_flat and not in_region:
            v_start = v[i]
            in_region = True
        elif not is_flat and in_region:
            regions.append((v_start, v[i - 1]))
            in_region = False
    if in_region:
        regions.append((v_start, v[-1]))
    return regions


def main():
    t0 = time.time()
    out_dir = os.path.join(_DIR, cfg.get('output_dir', 'output'))
    os.makedirs(out_dir, exist_ok=True)

    benchmarks = {'BP1': BP1, 'MAP': MAP}
    results = {}

    print("=" * 80)
    print("  η(v) = d ln(σ/m) / d ln v   —   Velocity Slope Analysis")
    print("=" * 80)

    for label, bp in benchmarks.items():
        lam = bp['alpha'] * bp['m_chi_GeV'] / (bp['m_phi_MeV'] / 1000.0)
        print(f"\n  --- {label}: m_χ={bp['m_chi_GeV']:.2f} GeV, "
              f"m_φ={bp['m_phi_MeV']:.2f} MeV, α={bp['alpha']:.3e}, λ={lam:.1f} ---")

        sigma = compute_sigma_grid(bp)
        log_v = np.log(V_GRID)
        log_sigma = np.log(np.maximum(sigma, 1e-30))
        eta = log_slope(log_v, log_sigma)

        results[label] = {'sigma': sigma, 'eta': eta, 'lambda': lam}

        # η at reference velocities
        print(f"    {'v [km/s]':>10s}  {'σ/m [cm²/g]':>12s}  {'η':>8s}  {'Δσ/σ for ±20% Δv':>20s}")
        print("    " + "-" * 56)
        for name, v_ref in V_REFS.items():
            idx = np.argmin(np.abs(V_GRID - v_ref))
            s = sigma[idx]
            e = eta[idx]
            frac_err = abs(e) * 0.2  # ±20% velocity → fractional σ error
            print(f"    {v_ref:>10.0f}  {s:>12.4f}  {e:>8.2f}  ±{frac_err * 100:>5.1f}%")

        # Extrema of |η|
        abs_eta = np.abs(eta)
        i_max = np.argmax(abs_eta[5:-5]) + 5  # skip edges
        print(f"\n    Max |η| = {abs_eta[i_max]:.2f} at v = {V_GRID[i_max]:.1f} km/s")

        # Flat regions
        flat = find_flat_regions(V_GRID, eta, threshold=0.3)
        if flat:
            print(f"    Flat regions (|η| < 0.3):")
            for v_lo, v_hi in flat:
                print(f"      v ∈ [{v_lo:.1f}, {v_hi:.1f}] km/s")
        else:
            print(f"    No extended flat region — σ/m changes everywhere")

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors = {'BP1': 'C0', 'MAP': 'C3'}
    styles = {'BP1': '-', 'MAP': '-'}

    # Panel (a): σ/m(v)
    ax = axes[0]
    for label in benchmarks:
        lam = results[label]['lambda']
        ax.loglog(V_GRID, results[label]['sigma'], styles[label],
                  color=colors[label], lw=1.8,
                  label=f'{label} (λ={lam:.1f})')
    for name, v_ref in V_REFS.items():
        ax.axvline(v_ref, ls=':', color='gray', lw=0.7)
        ax.text(v_ref, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-3,
                f'{name}\n{v_ref}', ha='center', va='bottom', fontsize=7, color='gray')
    ax.set_ylabel(r'$\sigma_T / m_\chi$ [cm$^2$/g]')
    ax.legend(fontsize=9)
    ax.set_title(r'(a) Cross-section $\sigma/m(v)$')
    ax.grid(True, alpha=0.3)

    # Panel (b): η(v)
    ax = axes[1]
    for label in benchmarks:
        ax.semilogx(V_GRID, results[label]['eta'], styles[label],
                    color=colors[label], lw=1.5,
                    label=f'{label}')
    ax.axhline(0, ls='--', color='k', lw=0.5)
    ax.axhline(-2, ls=':', color='gray', lw=0.8, label=r'$\eta = -2$ (perturbative)')
    ax.axhline(-1, ls=':', color='orange', lw=0.8, label=r'$\eta = -1$ ($N_{{scatter}}$ extremum)')
    for name, v_ref in V_REFS.items():
        ax.axvline(v_ref, ls=':', color='gray', lw=0.7)
    ax.fill_between(V_GRID, -0.3, 0.3, alpha=0.1, color='green', label=r'$|\eta|<0.3$ (robust)')
    ax.set_ylabel(r'$\eta(v) = d\ln(\sigma/m) / d\ln v$')
    ax.set_ylim(-5, 3)
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    ax.set_title(r'(b) Logarithmic slope $\eta(v)$')
    ax.grid(True, alpha=0.3)

    # Panel (c): Robustness — fractional σ error for ±20% velocity uncertainty
    ax = axes[2]
    for label in benchmarks:
        frac_err = np.abs(results[label]['eta']) * 0.2 * 100  # percent
        ax.semilogx(V_GRID, frac_err, styles[label],
                    color=colors[label], lw=1.5, label=f'{label}')
    ax.axhline(20, ls='--', color='red', lw=0.8, label='20% σ/m error')
    for name, v_ref in V_REFS.items():
        ax.axvline(v_ref, ls=':', color='gray', lw=0.7)
    ax.set_ylabel(r'$\Delta(\sigma/m) / (\sigma/m)$  [%] for $\pm 20\%$ $\Delta v$')
    ax.set_xlabel(r'$v_{\rm rel}$ [km/s]')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.set_title(r'(c) Robustness map: propagated velocity uncertainty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'velocity_slope.{ext}'), dpi=200)
    plt.close()
    print(f"\n  Saved: output/velocity_slope.png")
    print(f"  Saved: output/velocity_slope.pdf")

    # ── CSV ──
    csv_path = os.path.join(out_dir, 'velocity_slope.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['v_km_s']
        for label in benchmarks:
            header += [f'sigma_m_{label}', f'eta_{label}']
        writer.writerow(header)
        for i, v in enumerate(V_GRID):
            row = [f'{v:.4f}']
            for label in benchmarks:
                row.append(f'{results[label]["sigma"][i]:.6e}')
                row.append(f'{results[label]["eta"][i]:.4f}')
            writer.writerow(row)
    print(f"  Saved: output/velocity_slope.csv")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
    try:
        from tg_notify import notify
        notify("✅ velocity_slope done!")
    except Exception:
        pass
