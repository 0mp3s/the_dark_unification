#!/usr/bin/env python3
"""
V9 — v31_smart_scan.py
========================
"Smart Scan" — Cosmology as Compass, Galaxies as Test.

Strategy:
  For each (m_χ, m_φ) grid cell:
    1. Bisect α to find EXACT Ωh² = 0.120 (numerical Boltzmann)
    2. Compute σ/m(30) and σ/m(1000) with that α (VPM)
    3. Keep only SIDM-viable points per global_config.json (Elbert+15, KTY16, Harvey+15)

Every surviving point is a true "golden egg" — cosmologically correct AND
astrophysically viable, with zero KT approximation error.
"""
# === path setup (auto-generated) ================================
import sys as _sys, os as _os
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _os.path.join(_ROOT, 'core'))
DATA_DIR = _os.path.join(_ROOT, 'data')
# =================================================================

import sys, os, math, time, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from config_loader import load_config
from global_config import GC
from output_manager import timestamped_path
from run_logger import RunLogger

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# ==============================================================
#  Configuration — loaded from config.json if available
# ==============================================================
_CFG = load_config(__file__)
_GRID = _CFG.get("grid", {})
_BOLTZ = _CFG.get("boltzmann", {})

N_CHI = _GRID.get("n_chi", 20)
N_PHI = _GRID.get("n_phi", 30)

_m_chi_range = _GRID.get("m_chi_range_GeV", [10.0, 100.0])
_m_phi_range = _GRID.get("m_phi_range_GeV", [1.0e-3, 50.0e-3])

M_CHI_VALS = np.logspace(np.log10(_m_chi_range[0]), np.log10(_m_chi_range[1]), N_CHI)
M_PHI_VALS = np.logspace(np.log10(_m_phi_range[0]), np.log10(_m_phi_range[1]), N_PHI)

TARGET_OMEGA = _BOLTZ.get("target_omega_h2", 0.1200)
BISECT_RTOL  = _BOLTZ.get("bisect_rtol", 1e-4)
BISECT_MAX   = _BOLTZ.get("bisect_max_iter", 50)
ALPHA_LO     = _BOLTZ.get("alpha_range", [1.0e-5, 5.0e-2])[0]
ALPHA_HI     = _BOLTZ.get("alpha_range", [1.0e-5, 5.0e-2])[1]

# SIDM cuts — single source of truth: data/global_config.json (no fallback)
_cuts = GC.sidm_cuts()
SIGMA_M_30_LO   = _cuts["sigma_m_30_lo"]
SIGMA_M_30_HI   = _cuts["sigma_m_30_hi"]
SIGMA_M_1000_HI = _cuts["sigma_m_1000_hi"]

# ==============================================================
#  Worker function (runs in child process)
# ==============================================================
def process_cell(args):
    """Process one (m_chi, m_phi) cell.
    
    Returns dict with results, or None if α bisection fails.
    """
    m_chi, m_phi, i_chi, i_phi = args

    # Lazy imports inside worker (each process imports fresh)
    from v27_boltzmann_relic import solve_boltzmann, Y_to_omega_h2
    from v22_raw_scan import sigma_T_vpm

    def omega_from_alpha(alpha):
        sv0 = math.pi * alpha**2 / (4.0 * m_chi**2)
        _, Y_arr = solve_boltzmann(m_chi, sv0)
        return Y_to_omega_h2(Y_arr[-1], m_chi)

    # --- Bisect α for Ωh² = TARGET ---
    a_lo, a_hi = ALPHA_LO, ALPHA_HI

    try:
        om_lo = omega_from_alpha(a_lo)
        om_hi = omega_from_alpha(a_hi)
    except Exception:
        return None

    # Ωh² decreases with increasing α
    if not (om_lo > TARGET_OMEGA > om_hi):
        return None  # bracket doesn't straddle target

    alpha_sol = None
    omega_sol = None
    for _ in range(BISECT_MAX):
        a_mid = math.sqrt(a_lo * a_hi)  # geometric mean (log-spaced)
        om_mid = omega_from_alpha(a_mid)
        if om_mid > TARGET_OMEGA:
            a_lo = a_mid
        else:
            a_hi = a_mid
        if abs(om_mid - TARGET_OMEGA) / TARGET_OMEGA < BISECT_RTOL:
            alpha_sol = a_mid
            omega_sol = om_mid
            break

    if alpha_sol is None:
        return None

    # --- Compute σ/m at v=30 and v=1000 ---
    sm_30  = sigma_T_vpm(m_chi, m_phi, alpha_sol, 30.0)
    sm_1000 = sigma_T_vpm(m_chi, m_phi, alpha_sol, 1000.0)

    lam = alpha_sol * m_chi / m_phi

    return {
        'i_chi': i_chi,
        'i_phi': i_phi,
        'm_chi_GeV': m_chi,
        'm_phi_GeV': m_phi,
        'alpha': alpha_sol,
        'omega_h2': omega_sol,
        'lambda': lam,
        'sigma_m_30': sm_30,
        'sigma_m_1000': sm_1000,
    }


# ==============================================================
#  Main
# ==============================================================
if __name__ == '__main__':
    _rl = RunLogger(
        script="relic_density/smart_scan.py",
        stage="1 - Relic Density",
        params={"n_chi": N_CHI, "n_phi": N_PHI, "target_omega": TARGET_OMEGA,
                "sigma_m_30_lo": SIGMA_M_30_LO, "sigma_m_30_hi": SIGMA_M_30_HI,
                "sigma_m_1000_hi": SIGMA_M_1000_HI},
    )
    _rl.__enter__()
    t_start = time.time()
    total_cells = N_CHI * N_PHI

    print("=" * 70)
    print("  V9 — v31 Smart Scan")
    print("  Cosmology as Compass, Galaxies as Test")
    print("=" * 70)
    print(f"  Grid: {N_CHI} × {N_PHI} = {total_cells} cells")
    print(f"  m_χ ∈ [{M_CHI_VALS[0]:.1f}, {M_CHI_VALS[-1]:.1f}] GeV")
    print(f"  m_φ ∈ [{M_PHI_VALS[0]*1e3:.1f}, {M_PHI_VALS[-1]*1e3:.1f}] MeV")
    print(f"  Target Ωh² = {TARGET_OMEGA}")
    print(f"  SIDM cuts: σ/m(30) ∈ [{SIGMA_M_30_LO}, {SIGMA_M_30_HI}], σ/m(1000) < {SIGMA_M_1000_HI}")
    print(f"  α bracket: [{ALPHA_LO:.0e}, {ALPHA_HI:.0e}]")
    print()

    # Build task list
    tasks = []
    for i_chi, mc in enumerate(M_CHI_VALS):
        for i_phi, mp in enumerate(M_PHI_VALS):
            tasks.append((mc, mp, i_chi, i_phi))

    # --- JIT warmup in main process (populates Numba cache for children) ---
    print("  Warming up Numba JIT cache...")
    from v22_raw_scan import sigma_T_vpm
    _ = sigma_T_vpm(10.0, 5e-3, 5e-4, 30.0)
    print("  JIT warm-up done.\n")

    # --- Parallel execution ---
    n_workers = min(10, os.cpu_count() or 4)
    print(f"  Launching {n_workers} worker processes...")
    print()

    all_results = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(process_cell, t): t for t in tasks}

        for future in as_completed(future_map):
            done += 1
            if done % 60 == 0 or done == total_cells:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_cells - done) / rate if rate > 0 else 0
                print(f"  ... {done}/{total_cells} done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            result = future.result()
            if result is not None:
                all_results.append(result)

    t_bisect = time.time() - t_start
    print(f"\n  Bisection complete: {len(all_results)}/{total_cells} cells converged ({t_bisect:.1f}s)")

    # --- Apply SIDM filter ---
    viable = []
    for r in all_results:
        if (SIGMA_M_30_LO <= r['sigma_m_30'] <= SIGMA_M_30_HI and
                r['sigma_m_1000'] < SIGMA_M_1000_HI):
            viable.append(r)

    # Sort by |Ωh² - 0.120| then by σ/m(1000) ascending
    viable.sort(key=lambda r: (abs(r['omega_h2'] - TARGET_OMEGA), r['sigma_m_1000']))

    print(f"  SIDM-viable: {len(viable)}/{len(all_results)}")

    # --- Display results ---
    print("\n" + "=" * 70)
    if len(viable) == 0:
        print("  ❌ NO VIABLE POINTS FOUND on this grid.")
        print("  Consider refining the grid or relaxing cuts.")

        # Show near-misses
        near = [r for r in all_results if r['sigma_m_30'] >= 0.5 and r['sigma_m_1000'] < 0.15]
        near.sort(key=lambda r: r['sigma_m_1000'])
        if near:
            print(f"\n  NEAR-MISSES (σ/m(30) ≥ 0.5, σ/m(1000) < 0.15): {len(near)}")
            print("  {:>8s}  {:>8s}  {:>10s}  {:>8s}  {:>10s}  {:>10s}  {:>6s}".format(
                "m_χ", "m_φ", "α", "Ωh²", "σ/m(30)", "σ/m(1000)", "λ"))
            print("  " + "-" * 68)
            for r in near[:15]:
                print("  {:8.3f}  {:8.3f}  {:10.4e}  {:8.4f}  {:10.4f}  {:10.4f}  {:6.2f}".format(
                    r['m_chi_GeV'], r['m_phi_GeV']*1e3, r['alpha'],
                    r['omega_h2'], r['sigma_m_30'], r['sigma_m_1000'], r['lambda']))
    else:
        print(f"  ✅ {len(viable)} GOLDEN POINTS FOUND!")
        print("=" * 70)
        n_show = min(10, len(viable))
        print(f"\n  TOP-{n_show} (sorted by σ/m(1000) ascending):")
        print("  {:>3s}  {:>8s}  {:>8s}  {:>10s}  {:>8s}  {:>10s}  {:>10s}  {:>6s}".format(
            "#", "m_χ", "m_φ", "α", "Ωh²", "σ/m(30)", "σ/m(1000)", "λ"))
        print("  " + "-" * 72)
        for i, r in enumerate(viable[:n_show]):
            print("  {:3d}  {:8.3f}  {:8.3f}  {:10.4e}  {:8.4f}  {:10.4f}  {:10.4f}  {:6.2f}".format(
                i+1, r['m_chi_GeV'], r['m_phi_GeV']*1e3, r['alpha'],
                r['omega_h2'], r['sigma_m_30'], r['sigma_m_1000'], r['lambda']))

    # --- Save ALL results to CSV (including non-viable, for diagnostics) ---
    all_results.sort(key=lambda r: (r['m_chi_GeV'], r['m_phi_GeV']))
    _OUT = _CFG.get("output", {})
    _csv_all_rel = _OUT.get("all_relic_csv") or str(timestamped_path("v31_all_relic_points"))
    csv_all = _csv_all_rel if os.path.isabs(_csv_all_rel) else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _csv_all_rel))
    with open(csv_all, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m_chi_GeV', 'm_phi_MeV', 'alpha', 'omega_h2',
                         'lambda', 'sigma_m_30', 'sigma_m_1000', 'sidm_viable'])
        for r in all_results:
            is_viable = (SIGMA_M_30_LO <= r['sigma_m_30'] <= SIGMA_M_30_HI and
                         r['sigma_m_1000'] < SIGMA_M_1000_HI)
            writer.writerow([
                f"{r['m_chi_GeV']:.6f}", f"{r['m_phi_GeV'] * 1000:.6e}",
                f"{r['alpha']:.6e}", f"{r['omega_h2']:.6f}",
                f"{r['lambda']:.4f}", f"{r['sigma_m_30']:.6f}",
                f"{r['sigma_m_1000']:.6f}", int(is_viable)])
    print(f"\n  Saved {len(all_results)} points → {csv_all}")

    # --- Save viable-only CSV ---
    if viable:
        _csv_viable_rel = _OUT.get("viable_csv") or str(timestamped_path("v31_true_viable_points"))
        csv_viable = _csv_viable_rel if os.path.isabs(_csv_viable_rel) else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _csv_viable_rel))
        with open(csv_viable, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['m_chi_GeV', 'm_phi_MeV', 'alpha', 'omega_h2',
                             'lambda', 'sigma_m_30', 'sigma_m_1000'])
            for r in viable:
                writer.writerow([
                    f"{r['m_chi_GeV']:.6f}", f"{r['m_phi_GeV'] * 1000:.6e}",
                    f"{r['alpha']:.6e}", f"{r['omega_h2']:.6f}",
                    f"{r['lambda']:.4f}", f"{r['sigma_m_30']:.6f}",
                    f"{r['sigma_m_1000']:.6f}"])
        print(f"  Saved {len(viable)} viable → {csv_viable}")
        _rl.add_output(csv_viable)

    _rl.set_n_viable(len(viable))
    _rl.add_output(csv_all)
    _rl.__exit__(None, None, None)
    t_total = time.time() - t_start
    print(f"\n  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print("=" * 70)

    try:
        import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
        from tg_notify import notify
        notify(f"✅ smart_scan done!\nviable={len(viable)} / {len(all_results)}\nelapsed={t_total:.0f}s ({t_total/60:.1f} min)")
    except Exception:
        pass
