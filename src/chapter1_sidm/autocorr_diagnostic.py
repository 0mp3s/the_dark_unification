#!/usr/bin/env python3
"""
stats_mcmc/autocorr_diagnostic.py
=================================
MCMC convergence diagnostic: integrated autocorrelation time τ_int.

Loads the saved flat chain (v38_mcmc_samples.npy), reshapes to
(N_steps, N_walkers, N_dim), and computes τ_int per parameter
using emcee.autocorr.integrated_time().

Reports:
  - τ_int for each parameter
  - N_eff = N_steps × N_walkers / τ_max
  - N_steps / τ_int (should be > 50 for reliable posteriors)
  - Convergence diagnostic plot (τ vs chain length)

Reference: Foreman-Mackey+ (2013), Goodman & Weare (2010)
"""
import os, sys
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_DIR, '..', 'core')))
from output_manager import get_latest, _MCMC_ARCHIVE
OUT_DIR = os.path.join(_DIR, 'output')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import emcee

PARAM_LABELS = [r"$\log_{10}(m_\chi/\mathrm{GeV})$",
                r"$\log_{10}(m_\phi/\mathrm{MeV})$",
                r"$\log_{10}\alpha$"]
PARAM_NAMES = ["log10_m_chi", "log10_m_phi", "log10_alpha"]

N_WALKERS = 32   # must match the run


def main():
    npy_path = str(get_latest("v38_mcmc_samples", ".npy", _MCMC_ARCHIVE))

    flat = np.load(npy_path)
    n_total, ndim = flat.shape
    n_steps = n_total // N_WALKERS
    chain = flat.reshape(n_steps, N_WALKERS, ndim)

    print("=" * 72)
    print("  MCMC Convergence Diagnostic — Autocorrelation Time")
    print("=" * 72)
    print(f"\n  Chain shape: ({n_steps} steps) × ({N_WALKERS} walkers) × ({ndim} params)")
    print(f"  Total samples: {n_total:,}")

    # ── Compute final τ_int ──
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except emcee.autocorr.AutocorrError as e:
        print(f"\n  WARNING: {e}")
        tau = emcee.autocorr.integrated_time(chain, quiet=True, tol=0)

    print(f"\n  {'Parameter':<25s}  {'τ_int':>8s}  {'N/τ':>8s}  {'N_eff':>10s}  {'Status':>8s}")
    print("  " + "-" * 65)
    for i in range(ndim):
        n_over_tau = n_steps / tau[i]
        n_eff = n_total / tau[i]
        status = "PASS" if n_over_tau > 50 else "MARGINAL" if n_over_tau > 20 else "FAIL"
        print(f"  {PARAM_NAMES[i]:<25s}  {tau[i]:8.1f}  {n_over_tau:8.1f}  {n_eff:10.0f}  {status:>8s}")

    tau_max = np.max(tau)
    n_eff_min = n_total / tau_max
    n_over_tau_min = n_steps / tau_max

    print(f"\n  Max τ_int = {tau_max:.1f} steps")
    print(f"  Min N_eff = {n_eff_min:.0f} independent samples")
    print(f"  N_steps / τ_max = {n_over_tau_min:.1f}  (recommend > 50)")
    if n_over_tau_min > 50:
        print("  → CONVERGED: chain length is sufficient")
    elif n_over_tau_min > 20:
        print("  → MARGINAL: posteriors are usable but a longer chain is recommended")
    else:
        print("  → NOT CONVERGED: need a longer chain")

    # ── τ vs chain length (convergence diagnostic plot) ──
    # Compute τ for increasing sub-chains to show convergence
    check_steps = np.arange(100, n_steps + 1, 50)
    tau_vs_n = np.zeros((len(check_steps), ndim))

    for idx, n in enumerate(check_steps):
        sub_chain = chain[:n, :, :]
        try:
            tau_vs_n[idx] = emcee.autocorr.integrated_time(sub_chain, tol=0, quiet=True)
        except Exception:
            tau_vs_n[idx] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#E91E63', '#2196F3', '#4CAF50']
    for i in range(ndim):
        ax1.plot(check_steps, tau_vs_n[:, i], color=colors[i], lw=2,
                 label=f"{PARAM_NAMES[i]}: τ={tau[i]:.1f}")
    ax1.plot(check_steps, check_steps / 50.0, 'k--', lw=1, alpha=0.5,
             label='N/50 threshold')
    ax1.set_xlabel('Chain length N (steps)', fontsize=12)
    ax1.set_ylabel(r'$\hat{\tau}_{\mathrm{int}}$ (steps)', fontsize=12)
    ax1.set_title('Autocorrelation time vs chain length', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Autocorrelation function for each parameter ──
    max_lag = min(500, n_steps // 2)
    for i in range(ndim):
        # Mean across walkers, then compute ACF
        mean_chain = np.mean(chain[:, :, i], axis=1)
        mean_chain -= np.mean(mean_chain)
        acf = np.correlate(mean_chain, mean_chain, mode='full')
        acf = acf[len(acf)//2:]
        acf /= acf[0]
        ax2.plot(np.arange(max_lag), acf[:max_lag], color=colors[i], lw=1.5,
                 label=PARAM_NAMES[i], alpha=0.8)

    ax2.axhline(0, color='gray', ls='-', lw=0.5)
    ax2.axhline(0.05, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax2.set_xlabel('Lag (steps)', fontsize=12)
    ax2.set_ylabel('Autocorrelation', fontsize=12)
    ax2.set_title('Autocorrelation function (walker-averaged)', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_lag)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'v38_autocorr_diagnostic.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {fig_path}")
    print("  Done.")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
        from tg_notify import notify
        notify("\u2705 autocorr_diagnostic done!")
    except Exception:
        pass
