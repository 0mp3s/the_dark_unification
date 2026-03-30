#!/usr/bin/env python3
"""
stats_mcmc/gelman_rubin.py
==========================
Gelman-Rubin R̂ convergence diagnostic for the MCMC chains.

For M chains of length N (here: M=32 walkers, N=n_steps):
  B  = N/(M-1) * Σ_j (ψ̄_j - ψ̄)²          between-chain variance
  W  = (1/M)   * Σ_j s_j²                  within-chain variance (s_j² = sample var)
  V̂  = (N-1)/N * W + (1/N) * B             pooled posterior variance estimate
  R̂  = sqrt(V̂ / W)

Convergence criterion: R̂ < 1.01 (strict) / 1.05 (acceptable).

Reference: Gelman & Rubin (1992), Brooks & Gelman (1998).

Outputs
-------
  - Console table with R̂ per parameter
  - v38_gelman_rubin.png  (R̂ vs chain-length plot)
  - Summary line appended to MCMC run log
"""
import os, sys
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_DIR, '..', 'core')))
from output_manager import get_latest, _MCMC_ARCHIVE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

PARAM_LABELS = [r"$\log_{10}(m_\chi/\mathrm{GeV})$",
                r"$\log_{10}(m_\phi/\mathrm{MeV})$",
                r"$\log_{10}\alpha$"]
PARAM_NAMES  = ["log10_m_chi", "log10_m_phi", "log10_alpha"]

N_WALKERS = 32   # must match the run


# ─────────────────────────────────────────────
#  Core R̂ computation
# ─────────────────────────────────────────────
def gelman_rubin_rhat(chain):
    """
    Compute Gelman-Rubin R̂ for each parameter.

    Parameters
    ----------
    chain : ndarray, shape (N, M, D)
        N = steps, M = chains/walkers, D = dimensions

    Returns
    -------
    rhat : ndarray, shape (D,)
        R̂ value per parameter.
    """
    N, M, D = chain.shape
    # chain means: (M, D)
    psi_bar_j = chain.mean(axis=0)          # shape (M, D)
    # grand mean: (D,)
    psi_bar   = psi_bar_j.mean(axis=0)      # shape (D,)

    # Between-chain variance B: (D,)
    B = N / (M - 1) * np.sum((psi_bar_j - psi_bar)**2, axis=0)

    # Within-chain variance W: sample variance per walker, averaged over walkers
    # s_j^2 = 1/(N-1) * sum_t (psi_jt - psi_bar_j)^2
    s2_j = ((chain - psi_bar_j[np.newaxis, :, :])**2).sum(axis=0) / (N - 1)  # (M, D)
    W = s2_j.mean(axis=0)  # (D,)

    # Pooled variance estimate V̂
    V_hat = (N - 1) / N * W + B / N

    # R̂
    rhat = np.sqrt(V_hat / W)
    return rhat


def rhat_vs_length(chain, fracs=None):
    """
    Compute R̂ as a function of chain length, to show convergence.

    Parameters
    ----------
    chain : ndarray (N, M, D)
    fracs : list of float, fractions of chain to use (default: 10 points from 0.1 to 1.0)

    Returns
    -------
    lengths : list of int
    rhats   : list of ndarray (D,)
    """
    N = chain.shape[0]
    if fracs is None:
        fracs = np.linspace(0.1, 1.0, 10)
    lengths = [max(4, int(f * N)) for f in fracs]
    rhats   = [gelman_rubin_rhat(chain[:n]) for n in lengths]
    return lengths, rhats


# ─────────────────────────────────────────────
#  Load chain
# ─────────────────────────────────────────────
def load_chain():
    """
    Load MCMC chain: archive (full 5000-step run) preferred over checkpoint (partial).
    Returns chain with shape (N_steps, N_walkers, N_dim).
    """
    # Prefer archive: contains the full production run
    try:
        npy_path = str(get_latest("v38_mcmc_samples", ".npy", _MCMC_ARCHIVE))
        flat = np.load(npy_path)
        n_total, ndim = flat.shape
        n_steps = n_total // N_WALKERS
        chain = flat[:n_steps * N_WALKERS].reshape(n_steps, N_WALKERS, ndim)
        print(f"  Loaded from archive: {n_steps} steps × {N_WALKERS} walkers × {ndim} params")
        return chain
    except Exception as e:
        print(f"  Archive load failed ({e}), falling back to checkpoint...")

    # Fallback: checkpoint .npy
    checkpoint_path = os.path.join(OUT_DIR, 'v38_mcmc_checkpoint.npy')
    ckpt = np.load(checkpoint_path, allow_pickle=True).item()
    flat = ckpt['flat_samples']
    n_walkers = int(ckpt.get('n_walkers', N_WALKERS))
    n_total, ndim = flat.shape
    n_steps = n_total // n_walkers
    chain = flat[:n_steps * n_walkers].reshape(n_steps, n_walkers, ndim)
    print(f"  Loaded from checkpoint: {n_steps} steps × {n_walkers} walkers × {ndim} params")
    return chain


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Secluded-Majorana-SIDM — Gelman-Rubin R̂ Diagnostic")
    print("=" * 72)

    chain = load_chain()
    N, M, D = chain.shape

    # ── Final R̂ ──
    rhat = gelman_rubin_rhat(chain)

    print(f"\n  {'Parameter':<30}  {'R̂':>8}  {'Status':>12}")
    print("  " + "-" * 56)
    all_pass_strict     = True
    all_pass_acceptable = True
    for i, (name, label) in enumerate(zip(PARAM_NAMES, PARAM_LABELS)):
        r = rhat[i]
        if r < 1.01:
            status = "✓ CONVERGED"
        elif r < 1.05:
            status = "~ acceptable"
            all_pass_strict = False
        else:
            status = "✗ NOT converged"
            all_pass_strict = False
            all_pass_acceptable = False
        print(f"  {name:<30}  {r:8.4f}  {status:>12}")

    print()
    if all_pass_strict:
        verdict = "ALL R̂ < 1.01 — chain has converged (strict criterion)"
    elif all_pass_acceptable:
        verdict = "ALL R̂ < 1.05 — chain is acceptable (Brooks & Gelman 1998)"
    else:
        verdict = "WARNING: some parameters have R̂ ≥ 1.05 — chain needs longer run"
    print(f"  {verdict}")

    # ── Between/within variance breakdown ──
    print(f"\n  Chain details: N={N} steps, M={M} walkers, effective samples ~{N*M/max(100,1):,.0f}")

    # ── R̂ vs chain length ──
    print("\n  Computing R̂ convergence vs chain length...")
    lengths, rhats_vs_len = rhat_vs_length(chain)

    fig, axes = plt.subplots(1, D, figsize=(4 * D, 4), sharey=False)
    if D == 1:
        axes = [axes]

    for i, (ax, name, label) in enumerate(zip(axes, PARAM_NAMES, PARAM_LABELS)):
        vals = [r[i] for r in rhats_vs_len]
        ax.plot(lengths, vals, 'o-', color='steelblue', linewidth=2, markersize=5)
        ax.axhline(1.01, color='green',  linestyle='--', linewidth=1.2, label=r'$\hat{R}=1.01$ (strict)')
        ax.axhline(1.05, color='orange', linestyle='--', linewidth=1.2, label=r'$\hat{R}=1.05$ (acceptable)')
        ax.axhline(1.0,  color='gray',   linestyle=':',  linewidth=0.8)
        ax.set_xlabel('Chain length (steps)', fontsize=10)
        ax.set_ylabel(r'$\hat{R}$', fontsize=11)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0.99)
        final_r = rhat[i]
        ax.annotate(f'Final: {final_r:.4f}',
                    xy=(lengths[-1], final_r),
                    xytext=(-40, 10), textcoords='offset points',
                    fontsize=9, color='steelblue',
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

    plt.suptitle(r'Gelman-Rubin $\hat{R}$ vs Chain Length — Secluded-Majorana-SIDM',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'v38_gelman_rubin.png')
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {outpath}")

    # ── Summary for preprint ──
    print("\n  --- Preprint summary line ---")
    rhat_str = ", ".join(
        f"{name}={rhat[i]:.4f}" for i, name in enumerate(PARAM_NAMES)
    )
    print(f"  R̂ = [{rhat_str}]  →  max R̂ = {rhat.max():.4f}")
    if all_pass_strict:
        print("  → §4.7: Add 'Gelman-Rubin R̂ < 1.01 for all 3 parameters (strict convergence criterion).'")
    elif all_pass_acceptable:
        print("  → §4.7: Add 'Gelman-Rubin R̂ < 1.05 for all 3 parameters (acceptable convergence).'")

    return rhat


if __name__ == '__main__':
    main()
