#!/usr/bin/env python3
"""
stats_mcmc/run_mcmc.py  (based on V10/v38_mcmc_contours.py)
============================================================
MCMC posterior sampling and corner-plot generation for the Majorana-scalar
SIDM model (m_chi, m_phi, alpha), using emcee and the VPM solver.

Fits the 13 observational data points from KTY16/KKPY17/Randall+08/Harvey+15/Elbert+15.
Produces: output/v38_corner.png, output/v38_mcmc_chains.png,
          output/v38_lambda_posterior.png, output/v38_mcmc_samples.csv
"""
import sys, os, time, math
import numpy as np
import multiprocessing as mp_lib

# ---------- path bootstrap ----------
import sys as _sys, os as _os
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _os.path.join(_ROOT, 'core'))
DATA_DIR = _os.path.join(_ROOT, 'data')
# ------------------------------------

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
os.environ['PYTHONIOENCODING'] = 'utf-8'

import emcee
import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config_loader import load_config
from global_config import GC
from v22_raw_scan import sigma_T_vpm
from output_manager import get_latest, timestamped_path, _MCMC_ARCHIVE
from run_logger import RunLogger

# Warm up JIT
sigma_T_vpm(20.0, 10e-3, 1e-3, 100.0)

_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================
#  Load configuration
# ================================================================
cfg = load_config(__file__)

# ================================================================
#  Multiprocessing support
# ================================================================
_solver = None

def _init_worker():
    """Each worker imports and warms up the JIT solver."""
    global _solver
    import sys, os
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    core = os.path.join(root, 'core')
    if core not in sys.path:
        sys.path.insert(0, core)
    from v22_raw_scan import sigma_T_vpm
    _solver = sigma_T_vpm
    _solver(20.0, 10e-3, 1e-3, 100.0)  # warm up


def _log_prob_worker(theta):
    """Worker-safe log_prob using the per-process solver."""
    log_m_chi, log_m_phi_MeV, log_alpha = theta
    # Prior check
    if not (np.all(theta >= LO) and np.all(theta <= HI)):
        return -np.inf
    m_chi = 10.0 ** log_m_chi
    m_phi_MeV = 10.0 ** log_m_phi_MeV
    m_phi_GeV = m_phi_MeV / 1000.0
    alpha = 10.0 ** log_alpha
    chi2 = 0.0
    for name, v, central, lo, hi, ref in OBSERVATIONS:
        try:
            theory = _solver(m_chi, m_phi_GeV, alpha, float(v))
        except Exception:
            return -np.inf
        if np.isnan(theory) or np.isinf(theory) or theory < 0:
            return -np.inf
        # One-sided upper limits (lo == 0): no penalty when theory < hi
        if lo == 0.0 and theory <= hi:
            continue
        if theory >= central:
            sigma = hi - central if hi > central else 0.5 * central
        else:
            sigma = central - lo if central > lo else 0.5 * central
        if sigma <= 0:
            sigma = 0.5 * max(central, 0.01)
        chi2 += ((theory - central) / sigma) ** 2
    return -0.5 * chi2

# ================================================================
#  Observational data (same 13 systems as chi2_fit.py)
# ================================================================
OBSERVATIONS = GC.observations_as_tuples()

N_DATA = len(OBSERVATIONS)

# ================================================================
#  Prior bounds (log10 space) — configurable
# ================================================================
_priors = cfg.get("priors", {})
_m_chi_lo  = _priors.get("m_chi_GeV_min", 5.0)
_m_chi_hi  = _priors.get("m_chi_GeV_max", 200.0)
_m_phi_lo  = _priors.get("m_phi_MeV_min", 3.0)
_m_phi_hi  = _priors.get("m_phi_MeV_max", 30.0)
_alpha_lo  = _priors.get("alpha_min", 1e-5)
_alpha_hi  = _priors.get("alpha_max", 0.05)

LO = np.array([np.log10(_m_chi_lo), np.log10(_m_phi_lo), np.log10(_alpha_lo)])
HI = np.array([np.log10(_m_chi_hi), np.log10(_m_phi_hi), np.log10(_alpha_hi)])

PARAM_LABELS = [r"$\log_{10}(m_\chi/{\rm GeV})$",
                r"$\log_{10}(m_\phi/{\rm MeV})$",
                r"$\log_{10}\alpha$"]

NDIM = 3


# ================================================================
#  chi2 computation
# ================================================================
def compute_chi2(m_chi, m_phi_GeV, alpha):
    """Return chi2 for given physical parameters. NaN if VPM fails."""
    chi2 = 0.0
    for name, v, central, lo, hi, ref in OBSERVATIONS:
        try:
            theory = sigma_T_vpm(m_chi, m_phi_GeV, alpha, float(v))
        except Exception:
            return np.nan
        if np.isnan(theory) or np.isinf(theory) or theory < 0:
            return np.nan
        # One-sided upper limits (lo == 0): no penalty when theory < hi
        if lo == 0.0 and theory <= hi:
            continue
        if theory >= central:
            sigma = hi - central if hi > central else 0.5 * central
        else:
            sigma = central - lo if central > lo else 0.5 * central
        if sigma <= 0:
            sigma = 0.5 * max(central, 0.01)
        chi2 += ((theory - central) / sigma) ** 2
    return chi2


# ================================================================
#  Log-probability for emcee
# ================================================================
def log_prior(theta):
    if np.all(theta >= LO) and np.all(theta <= HI):
        return 0.0
    return -np.inf


def log_likelihood(theta):
    log_m_chi, log_m_phi_MeV, log_alpha = theta
    m_chi = 10.0 ** log_m_chi
    m_phi_MeV = 10.0 ** log_m_phi_MeV
    m_phi_GeV = m_phi_MeV / 1000.0
    alpha = 10.0 ** log_alpha
    c2 = compute_chi2(m_chi, m_phi_GeV, alpha)
    if np.isnan(c2) or np.isinf(c2):
        return -np.inf
    return -0.5 * c2


def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ================================================================
#  MCMC sampling
# ================================================================
def run_mcmc():
    t0_total = time.time()
    hdr = "=" * 72

    print(hdr)
    print("  Secluded-Majorana-SIDM — MCMC Posterior Sampling")
    print(f"  {NDIM}D parameter space, {N_DATA} observational constraints")
    print(hdr)

    # -- Configuration --
    _mcmc = cfg.get("mcmc", {})
    N_WALKERS = _mcmc.get("n_walkers", 32)
    N_BURN = _mcmc.get("n_burn", 300)
    N_STEPS = _mcmc.get("n_production", 2000)
    ncpu = os.cpu_count() or 4
    nworkers = _mcmc.get("n_workers", max(1, ncpu - 2))
    _rl = RunLogger(
        script="stats_mcmc/run_mcmc.py",
        stage="4 - MCMC",
        params={"n_walkers": N_WALKERS, "n_burn": N_BURN, "n_steps": N_STEPS, "ndim": NDIM},
    )
    _rl.__enter__()
    print(f"\n  Walkers: {N_WALKERS}")
    print(f"  Burn-in: {N_BURN}")
    print(f"  Production: {N_STEPS}")
    print(f"  Workers: {nworkers} (of {ncpu} cores)")
    print(f"  Total likelihood evals: {N_WALKERS * (N_BURN + N_STEPS):,}")

    # -- Initialize walkers around the best-fit region --
    import csv as csv_mod
    seeds = []

    # Add relic BPs
    csv_path = cfg.get("relic_bp_csv") or str(get_latest("v31_true_viable_points"))
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            mc = float(row['m_chi_GeV'])
            mp = float(row['m_phi_MeV'])
            al = float(row['alpha'])
            seeds.append([np.log10(mc), np.log10(mp), np.log10(al)])

    # Add v34 unconstrained best fit
    seeds.append([np.log10(100.0), np.log10(9.15), np.log10(2.84e-3)])

    seeds = np.array(seeds)
    print(f"\n  Seed points: {len(seeds)}")

    # Initialize walkers: scatter around seed points
    rng = np.random.default_rng(42)
    p0 = np.zeros((N_WALKERS, NDIM))
    for i in range(N_WALKERS):
        seed = seeds[i % len(seeds)]
        p0[i] = seed + rng.normal(0, 0.05, NDIM)
        p0[i] = np.clip(p0[i], LO + 0.01, HI - 0.01)

    # -- Verify initial points --
    print("\n  Verifying initial walker positions...")
    bad = 0
    for i in range(N_WALKERS):
        lp = log_prob(p0[i])
        if not np.isfinite(lp):
            bad += 1
    print(f"  {N_WALKERS - bad}/{N_WALKERS} walkers have finite log-prob")

    if bad > N_WALKERS // 2:
        print("  WARNING: too many bad walkers, adjusting...")
        best_seed = seeds[0]
        for i in range(N_WALKERS):
            for attempt in range(50):
                p0[i] = best_seed + rng.normal(0, 0.02, NDIM)
                p0[i] = np.clip(p0[i], LO + 0.01, HI - 0.01)
                if np.isfinite(log_prob(p0[i])):
                    break

    # -- Run sampler with multiprocessing --
    pool = mp_lib.Pool(nworkers, initializer=_init_worker)
    sampler = emcee.EnsembleSampler(N_WALKERS, NDIM, _log_prob_worker, pool=pool)

    # Burn-in
    print(f"\n  Running burn-in ({N_BURN} steps)...")
    t0 = time.time()
    state = sampler.run_mcmc(p0, N_BURN, progress=True)
    dt_burn = time.time() - t0
    print(f"  Burn-in done in {dt_burn:.0f}s ({N_BURN * N_WALKERS / dt_burn:.1f} evals/s)")

    accept_burn = np.mean(sampler.acceptance_fraction)
    print(f"  Mean acceptance fraction (burn-in): {accept_burn:.3f}")

    sampler.reset()

    # Production
    print(f"\n  Running production ({N_STEPS} steps)...")
    t0 = time.time()
    sampler.run_mcmc(state, N_STEPS, progress=True)
    dt_prod = time.time() - t0
    print(f"  Production done in {dt_prod:.0f}s ({N_STEPS * N_WALKERS / dt_prod:.1f} evals/s)")

    pool.close()
    pool.join()

    # -- Diagnostics --
    accept = np.mean(sampler.acceptance_fraction)
    print(f"\n  Mean acceptance fraction: {accept:.3f}")

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"  Autocorrelation times: {tau}")
        n_eff = N_STEPS * N_WALKERS / np.max(tau)
        print(f"  Effective samples: ~{n_eff:.0f}")
    except Exception as e:
        print(f"  Autocorrelation: could not estimate ({e})")
        tau = None

    # -- Extract samples --
    flat_samples = sampler.get_chain(flat=True)
    print(f"\n  Total samples: {flat_samples.shape[0]}")

    # Compute physical parameters + chi2 for summary
    log_probs = sampler.get_log_prob(flat=True)
    best_idx = np.argmax(log_probs)
    best_theta = flat_samples[best_idx]
    best_chi2 = -2.0 * log_probs[best_idx]

    print(f"\n  Best-fit (MAP):")
    print(f"    log10(m_chi/GeV) = {best_theta[0]:.4f}  =>  m_chi = {10**best_theta[0]:.2f} GeV")
    print(f"    log10(m_phi/MeV) = {best_theta[1]:.4f}  =>  m_phi = {10**best_theta[1]:.2f} MeV")
    print(f"    log10(alpha)     = {best_theta[2]:.4f}  =>  alpha = {10**best_theta[2]:.3e}")
    lam = 2 * (10**best_theta[0]) * (10**best_theta[2]) / (10**best_theta[1] / 1000.0)
    print(f"    lambda = {lam:.2f}")
    print(f"    chi2 = {best_chi2:.3f},  chi2/dof = {best_chi2 / (N_DATA - NDIM):.4f}")

    # Credible intervals (median + 68%)
    print(f"\n  Parameter constraints (median +/- 68% CI):")
    print(f"  {'Parameter':<30s}  {'Median':>10s}  {'lo (16%)':>10s}  {'hi (84%)':>10s}")
    print("  " + "-" * 65)
    phys_labels = ["m_chi [GeV]", "m_phi [MeV]", "alpha"]
    for i in range(NDIM):
        q = np.percentile(flat_samples[:, i], [16, 50, 84])
        lo_val = 10**q[0]
        med_val = 10**q[1]
        hi_val = 10**q[2]
        print(f"  {phys_labels[i]:<30s}  {med_val:>10.3f}  {lo_val:>10.3f}  {hi_val:>10.3f}")

    # Also report lambda
    lam_samples = (2 * 10**flat_samples[:, 2] * 10**flat_samples[:, 0]
                   / (10**flat_samples[:, 1] / 1000.0))
    q_lam = np.percentile(lam_samples, [16, 50, 84])
    print(f"  {'lambda (derived)':<30s}  {q_lam[1]:>10.2f}  {q_lam[0]:>10.2f}  {q_lam[2]:>10.2f}")

    # ================================================================
    #  Corner plot
    # ================================================================
    print("\n  Generating corner plot...")

    fig = corner.corner(
        flat_samples,
        labels=PARAM_LABELS,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 13},
        truths=best_theta,
        truth_color='red',
        levels=(0.68, 0.95),
        color='steelblue',
        plot_datapoints=True,
        datapoint_kwargs={"alpha": 0.01, "ms": 1},
    )

    fig.suptitle(
        f"Majorana-scalar SIDM: MCMC posterior ({N_WALKERS} walkers, "
        f"{N_STEPS} steps)\n"
        f"Best fit: $m_\\chi$={10**best_theta[0]:.1f} GeV, "
        f"$m_\\phi$={10**best_theta[1]:.1f} MeV, "
        f"$\\alpha$={10**best_theta[2]:.1e}, "
        f"$\\chi^2/\\nu$={best_chi2/(N_DATA-NDIM):.3f}",
        fontsize=11, y=1.02
    )

    fig_path = os.path.join(OUT_DIR, 'v38_corner.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Corner plot saved: {fig_path}")
    plt.close(fig)

    # ================================================================
    #  Chain trace plot (convergence diagnostic)
    # ================================================================
    print("  Generating chain trace plot...")
    chain = sampler.get_chain()

    fig2, axes2 = plt.subplots(NDIM, 1, figsize=(10, 7), sharex=True)
    for i in range(NDIM):
        ax = axes2[i]
        for w in range(min(N_WALKERS, 16)):
            ax.plot(chain[:, w, i], alpha=0.3, lw=0.5)
        ax.set_ylabel(PARAM_LABELS[i], fontsize=10)
        ax.axhline(best_theta[i], color='red', ls='--', lw=1, alpha=0.7)
    axes2[-1].set_xlabel("Step")
    axes2[0].set_title(f"MCMC Chain Traces (accept={accept:.3f})")

    fig2_path = os.path.join(OUT_DIR, 'v38_mcmc_chains.png')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"  Chain trace saved: {fig2_path}")
    plt.close(fig2)

    # ================================================================
    #  Lambda posterior histogram
    # ================================================================
    print("  Generating lambda posterior...")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.hist(lam_samples, bins=80, density=True, color='steelblue',
             alpha=0.7, edgecolor='navy', lw=0.3)
    ax3.axvline(q_lam[1], color='red', ls='-', lw=2,
                label=f'Median = {q_lam[1]:.1f}')
    ax3.axvline(q_lam[0], color='red', ls='--', lw=1,
                label=f'68% CI: [{q_lam[0]:.1f}, {q_lam[2]:.1f}]')
    ax3.axvline(q_lam[2], color='red', ls='--', lw=1)
    ax3.set_xlabel(r'$\lambda = 2\alpha m_\chi / m_\phi$', fontsize=13)
    ax3.set_ylabel('Posterior density', fontsize=12)
    ax3.set_title(r'Derived parameter $\lambda$ posterior')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    fig3_path = os.path.join(OUT_DIR, 'v38_lambda_posterior.png')
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"  Lambda posterior saved: {fig3_path}")
    plt.close(fig3)

    # ================================================================
    #  Save chains for reproducibility
    # ================================================================
    npy_path = str(timestamped_path("v38_mcmc_samples", ".npy", _MCMC_ARCHIVE))
    np.save(npy_path, flat_samples)
    print(f"\n  Samples saved: {npy_path} ({flat_samples.shape})")

    import csv as csv_mod
    csv_out = str(timestamped_path("v38_mcmc_samples", ".csv", _MCMC_ARCHIVE))
    with open(csv_out, 'w', newline='') as f:
        w = csv_mod.writer(f)
        w.writerow(['log10_m_chi_GeV', 'log10_m_phi_MeV', 'log10_alpha',
                     'm_chi_GeV', 'm_phi_MeV', 'alpha', 'lambda', 'log_prob'])
        for i in range(flat_samples.shape[0]):
            s = flat_samples[i]
            mc = 10**s[0]; mp = 10**s[1]; al = 10**s[2]
            lam_i = 2 * al * mc / (mp / 1000.0)
            w.writerow([f"{s[0]:.6f}", f"{s[1]:.6f}", f"{s[2]:.6f}",
                        f"{mc:.4f}", f"{mp:.4f}", f"{al:.6e}",
                        f"{lam_i:.4f}", f"{log_probs[i]:.4f}"])
    print(f"  CSV saved: {csv_out} ({flat_samples.shape[0]} rows)")

    # ================================================================
    #  Relic BPs on posterior
    # ================================================================
    print(f"\n  Relic BPs vs posterior:")
    print(f"  {'BP':>3s}  {'m_chi':>7s}  {'m_phi':>7s}  {'alpha':>10s}  {'lam':>6s}  {'chi2':>8s}  {'within 95%':>10s}")
    print("  " + "-" * 60)

    chi2_samples = -2.0 * log_probs
    c2_95 = np.percentile(chi2_samples, 95)

    for row_idx, seed in enumerate(seeds[:-1]):
        mc = 10**seed[0]
        mp_MeV = 10**seed[1]
        mp_GeV = mp_MeV / 1000.0
        al = 10**seed[2]
        lam_bp = 2 * al * mc / mp_GeV
        c2 = compute_chi2(mc, mp_GeV, al)
        within = "YES" if c2 <= c2_95 else "no"
        print(f"  {row_idx+1:3d}  {mc:7.1f}  {mp_MeV:7.2f}  {al:10.3e}  {lam_bp:6.2f}  "
              f"{c2:8.3f}  {within:>10s}")

    # ================================================================
    #  Summary
    # ================================================================
    dt_total = time.time() - t0_total
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  Total time: {dt_total/60:.1f} min ({dt_total/3600:.2f} h)")
    print(f"  Total samples: {flat_samples.shape[0]:,}")
    print(f"  Acceptance: {accept:.3f}")
    if tau is not None:
        print(f"  Max autocorr: {np.max(tau):.1f} steps")
    print(f"  Best chi2/dof: {best_chi2/(N_DATA-NDIM):.4f}")
    print(f"  Files: v38_corner.png, v38_mcmc_chains.png, v38_lambda_posterior.png")
    print(f"{'='*72}")
    _rl.add_output(fig_path)
    _rl.add_output(fig2_path)
    _rl.add_output(fig3_path)
    _rl.add_output(npy_path)
    _rl.add_output(csv_out)
    _rl.set_notes(f"accept={accept:.3f}, chi2/dof={best_chi2/(N_DATA-NDIM):.4f}")
    _rl.__exit__(None, None, None)


if __name__ == "__main__":
    run_mcmc()


if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
        from tg_notify import notify
        notify("\u2705 run_mcmc done!")
    except Exception:
        pass
