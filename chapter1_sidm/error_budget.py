#!/usr/bin/env python3
"""
V10 - error_budget.py
=========================
Systematic numerical error analysis for the VPM solver.

Three independent error components:
  1. Integrator error:   RK4 (fixed-step) vs scipy RK45 (adaptive, rtol=1e-10)
  2. Truncation error:   l_max  vs  l_max+5  vs  l_max+10  vs  l_max+20
  3. Prescription error:  x_min = 0.8*l/kappa  vs  l/kappa  vs  1.2*l/kappa

Tested on BP1 + MAP + BP17 (marginal) x 2 velocities (30, 1000 km/s).

Uses Numba JIT for truncation/prescription tests (fast),
scipy only for integrator ground truth comparison.
"""
import sys, os, math, time, json
import numpy as np
from numba import jit
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, os.path.join(_ROOT, 'core'))
from global_config import GC
from output_manager import get_latest

# ==============================================================
#  Constants (sourced from global_config.json)
# ==============================================================
_PC = GC.physical_constants()
GEV2_TO_CM2 = _PC["GEV2_to_cm2"]
GEV_IN_G    = _PC["GeV_in_g"]
C_KM_S      = _PC["c_km_s"]

# ==============================================================
#  Benchmarks — BP1 & MAP from global config, BP17 from relic CSV
# ==============================================================
def _load_benchmarks():
    """Load BP1, MAP from global_config.json and BP17 (marginal) from relic CSV."""
    bp1 = GC.benchmark("BP1")
    map_ = GC.benchmark("MAP")

    # BP17 = last row of the relic CSV (highest sigma_m_1000 among viable)
    import csv
    with open(str(get_latest("v31_true_viable_points")), "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    bp17 = rows[-1]  # BP17 is the last row (sorted by sigma_m_1000)

    return [
        {"name": "BP1 (relic, λ≈1.9)",
         "m_chi": bp1["m_chi_GeV"],
         "m_phi": bp1["m_phi_MeV"] * 1e-3,
         "alpha": bp1["alpha"]},
        {"name": "MAP (MCMC best-fit, λ≈48.6)",
         "m_chi": map_["m_chi_GeV"],
         "m_phi": map_["m_phi_MeV"] * 1e-3,
         "alpha": map_["alpha"]},
        {"name": "BP17 (marginal, σ/m(1000)≈0.099)",
         "m_chi": float(bp17["m_chi_GeV"]),
         "m_phi": float(bp17["m_phi_MeV"]) * 1e-3,
         "alpha": float(bp17["alpha"])},
    ]

BENCHMARKS = _load_benchmarks()
VELOCITIES = [30.0, 1000.0]

# ==============================================================
#  Numba Bessel functions (identical to v22)
# ==============================================================

@jit(nopython=True, cache=True)
def sph_jn_numba(l, z):
    if z < 1e-30:
        return 1.0 if l == 0 else 0.0
    j0 = math.sin(z) / z
    if l == 0: return j0
    j1 = math.sin(z) / (z * z) - math.cos(z) / z
    if l == 1: return j1
    j_prev, j_curr = j0, j1
    for n in range(1, l):
        j_next = (2 * n + 1) / z * j_curr - j_prev
        j_prev, j_curr = j_curr, j_next
        if abs(j_curr) < 1e-300: return 0.0
    return j_curr

@jit(nopython=True, cache=True)
def sph_yn_numba(l, z):
    if z < 1e-30: return -1e300
    y0 = -math.cos(z) / z
    if l == 0: return y0
    y1 = -math.cos(z) / (z * z) - math.sin(z) / z
    if l == 1: return y1
    y_prev, y_curr = y0, y1
    for n in range(1, l):
        y_next = (2 * n + 1) / z * y_curr - y_prev
        y_prev, y_curr = y_curr, y_next
        if abs(y_curr) > 1e200: return y_curr
    return y_curr

# ==============================================================
#  Numba VPM phase shift solver (identical to v22)
# ==============================================================

@jit(nopython=True, cache=True)
def _vpm_rhs(l, kappa, lam, x, delta):
    if x < 1e-20: return 0.0
    z = kappa * x
    if z < 1e-20: return 0.0
    jl = sph_jn_numba(l, z)
    nl = sph_yn_numba(l, z)
    j_hat = z * jl
    n_hat = -z * nl
    cd = math.cos(delta)
    sd = math.sin(delta)
    bracket = j_hat * cd - n_hat * sd
    if not math.isfinite(bracket): return 0.0
    pot = lam * math.exp(-x) / (kappa * x)
    val = pot * bracket * bracket
    return val if math.isfinite(val) else 0.0

@jit(nopython=True, cache=True)
def vpm_phase_shift(l, kappa, lam, x_max, N_steps, x_min_override=-1.0):
    """RK4 VPM solver with optional x_min override."""
    if lam < 1e-30 or kappa < 1e-30: return 0.0
    x_min = max(1e-5, 0.05 / (kappa + 0.01))
    if l > 0:
        x_barrier = l / kappa
        if x_barrier > x_min:
            x_min = x_barrier
    if x_min_override > 0:
        x_min = x_min_override
    h = (x_max - x_min) / N_steps
    delta = 0.0
    for i in range(N_steps):
        x = x_min + i * h
        k1 = _vpm_rhs(l, kappa, lam, x, delta)
        k2 = _vpm_rhs(l, kappa, lam, x + 0.5*h, delta + 0.5*h*k1)
        k3 = _vpm_rhs(l, kappa, lam, x + 0.5*h, delta + 0.5*h*k2)
        k4 = _vpm_rhs(l, kappa, lam, x + h, delta + h*k3)
        delta += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return delta

# ==============================================================
#  Flexible sigma_T (Numba) with overrides
# ==============================================================

@jit(nopython=True, cache=True)
def sigma_T_flex(m_chi, m_phi, alpha, v_km_s,
                 l_max_override=-1, N_steps_override=-1,
                 x_min_factor=1.0):
    """sigma_T with optional l_max, N_steps, and x_min overrides."""
    v = v_km_s / C_KM_S
    mu = m_chi / 2.0
    k = mu * v
    kappa = k / m_phi
    lam = alpha * m_chi / m_phi
    if kappa < 1e-15: return 0.0

    # Production defaults
    if kappa < 5:
        x_max = 50.0; N_steps = 4000
    elif kappa < 50:
        x_max = 80.0; N_steps = 8000
    else:
        x_max = 100.0; N_steps = 12000

    if N_steps_override > 0:
        N_steps = N_steps_override
    # Adaptive l_max: min of physical limit (l/kappa < x_max) and lambda limit
    l_max = min(max(3, min(int(kappa * x_max), int(kappa) + int(lam) + 20)), 500)
    if l_max_override > 0:
        l_max = l_max_override

    sigma_sum = 0.0
    for l in range(l_max + 1):
        x_min_ov = -1.0
        if x_min_factor != 1.0 and l > 0:
            x_min_ov = x_min_factor * l / kappa
        delta = vpm_phase_shift(l, kappa, lam, x_max, N_steps, x_min_ov)
        weight = 1.0 if l % 2 == 0 else 3.0
        contrib = weight * (2*l + 1) * math.sin(delta)**2
        sigma_sum += contrib

    sigma_GeV2 = 2.0 * math.pi * sigma_sum / (k * k)
    sigma_cm2 = sigma_GeV2 * GEV2_TO_CM2
    return sigma_cm2 / (m_chi * GEV_IN_G)

# ==============================================================
#  Scipy reference solver (for integrator comparison only)
# ==============================================================

def scipy_phase_shift(l, kappa, lam, x_max=80.0):
    x_min = max(1e-6, 0.05 / (kappa + 0.01))
    if l > 0:
        x_barrier = l / kappa
        if x_barrier > x_min:
            x_min = x_barrier
    def rhs(x, y):
        z = kappa * x
        if z < 1e-30: return [0.0]
        jl = spherical_jn(l, z)
        nl = spherical_yn(l, z)
        j_hat, n_hat = z * jl, -z * nl
        pot = lam * math.exp(-x) / (kappa * x)
        cd, sd = math.cos(y[0]), math.sin(y[0])
        b = j_hat * cd - n_hat * sd
        if not math.isfinite(b): return [0.0]
        v = pot * b * b
        return [v] if math.isfinite(v) else [0.0]
    sol = solve_ivp(rhs, [x_min, x_max], [0.0], method='RK45',
                    rtol=1e-10, atol=1e-13, max_step=0.5)
    return sol.y[0, -1] if sol.success else float('nan')

def scipy_sigma_T(m_chi, m_phi, alpha, v_km_s):
    """sigma_T with scipy (same l_max as production for fair comparison)."""
    v = v_km_s / C_KM_S
    mu = m_chi / 2.0
    k = mu * v
    kappa = k / m_phi
    lam = alpha * m_chi / m_phi
    if kappa < 1e-15: return 0.0
    x_max_ref = 50.0 if kappa < 5 else (80.0 if kappa < 50 else 100.0)
    l_max = min(max(3, min(int(kappa * x_max_ref), int(kappa) + int(lam) + 20)), 500)
    sigma_sum = 0.0
    for l in range(l_max + 1):
        delta = scipy_phase_shift(l, kappa, lam)
        weight = 1.0 if l % 2 == 0 else 3.0
        sigma_sum += weight * (2*l + 1) * math.sin(delta)**2
    sigma_GeV2 = 2.0 * math.pi * sigma_sum / (k * k)
    sigma_cm2 = sigma_GeV2 * GEV2_TO_CM2
    return sigma_cm2 / (m_chi * GEV_IN_G)

# ==============================================================
#  Helpers
# ==============================================================

def get_params(bm, v_km_s):
    v = v_km_s / C_KM_S
    mu = bm["m_chi"] / 2.0
    k = mu * v
    kappa = k / bm["m_phi"]
    lam = bm["alpha"] * bm["m_chi"] / bm["m_phi"]
    x_max_ref = 50.0 if kappa < 5 else (80.0 if kappa < 50 else 100.0)
    l_max = min(max(3, min(int(kappa * x_max_ref), int(kappa) + int(lam) + 20)), 500)
    return kappa, lam, l_max

# ==============================================================
#  Test 1: Integrator Error (RK4 N_steps convergence)
# ==============================================================

def test_integrator():
    print("=" * 75)
    print("TEST 1: Integrator Error — RK4 step-size convergence")
    print("  Compare N_steps = 1000, 2000, 4000 (default), 8000, 16000")
    print("  Also vs scipy RK45 (rtol=1e-10) as ground truth")
    print("=" * 75)
    print()

    N_vals = [1000, 2000, 4000, 8000, 16000]

    for bm in BENCHMARKS:
        print(f"  --- {bm['name']} ---")
        print(f"  m_chi={bm['m_chi']:.3f} GeV, m_phi={bm['m_phi']*1000:.3f} MeV, "
              f"alpha={bm['alpha']:.3e}")

        for v in VELOCITIES:
            kappa, lam, l_max = get_params(bm, v)
            print(f"\n    v = {v:.0f} km/s  (kappa={kappa:.3f}, l_max={l_max})")

            sig_scipy = scipy_sigma_T(bm["m_chi"], bm["m_phi"], bm["alpha"], v)

            print(f"    {'N_steps':>8}  {'sigma/m':>12}  {'vs scipy':>10}  {'vs prev':>10}")
            print(f"    {'':->8}  {'':->12}  {'':->10}  {'':->10}")

            prev = None
            for N in N_vals:
                sig = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v,
                                   N_steps_override=N)
                err_scipy = (sig - sig_scipy) / sig_scipy * 100 if sig_scipy > 0 else 0
                err_prev = ""
                if prev is not None and prev > 0:
                    err_prev = f"{(sig - prev)/prev*100:+.4f}%"
                prev = sig
                default = " <-- default" if N == 4000 else ""
                print(f"    {N:>8}  {sig:12.6f}  {err_scipy:+.4f}%  {err_prev:>10}{default}")

            print(f"    {'scipy':>8}  {sig_scipy:12.6f}  {'(ref)':>10}")
        print()

# ==============================================================
#  Test 2: Truncation Error (l_max convergence) — Numba fast
# ==============================================================

def test_truncation():
    print("=" * 75)
    print("TEST 2: Truncation Error — l_max convergence")
    print("  Compare l_max, l_max+5, l_max+10, l_max+20")
    print("=" * 75)
    print()

    for bm in BENCHMARKS:
        print(f"  --- {bm['name']} ---")
        for v in VELOCITIES:
            kappa, lam, l_max_default = get_params(bm, v)
            offsets = [0, 5, 10, 20]
            print(f"\n    v = {v:.0f} km/s  (kappa={kappa:.3f}, default l_max={l_max_default})")
            print(f"    {'l_max':>8}  {'sigma/m':>12}  {'vs default':>12}")
            print(f"    {'':->8}  {'':->12}  {'':->12}")

            sig_default = None
            for dl in offsets:
                lm = l_max_default + dl
                sig = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v,
                                   l_max_override=lm)
                if dl == 0:
                    sig_default = sig
                err = ""
                if sig_default is not None and sig_default > 0:
                    err = f"{(sig - sig_default)/sig_default*100:+.4f}%"
                tag = " <-- default" if dl == 0 else ""
                print(f"    {lm:>8}  {sig:12.6f}  {err:>12}{tag}", flush=True)
        print()

# ==============================================================
#  Test 3: Prescription Error (x_min variation) — Numba fast
# ==============================================================

def test_prescription():
    print("=" * 75)
    print("TEST 3: Prescription Error — x_min starting point variation")
    print("  Compare x_min factor = 0.8, 0.9, 1.0 (default), 1.1, 1.2")
    print("=" * 75)
    print()

    factors = [0.8, 0.9, 1.0, 1.1, 1.2]

    for bm in BENCHMARKS:
        print(f"  --- {bm['name']} ---")
        for v in VELOCITIES:
            kappa, lam, l_max = get_params(bm, v)
            print(f"\n    v = {v:.0f} km/s  (kappa={kappa:.3f}, l_max={l_max})")
            print(f"    {'x_min fac':>10}  {'sigma/m':>12}  {'vs default':>12}")
            print(f"    {'':->10}  {'':->12}  {'':->12}")

            sig_default = None
            for fac in factors:
                sig = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v,
                                   x_min_factor=fac)
                if fac == 1.0:
                    sig_default = sig
                err = ""
                if sig_default is not None and sig_default > 0:
                    err = f"{(sig - sig_default)/sig_default*100:+.4f}%"
                tag = " <-- default" if fac == 1.0 else ""
                print(f"    {fac:>10.1f}  {sig:12.6f}  {err:>12}{tag}", flush=True)
        print()

# ==============================================================
#  Summary Table
# ==============================================================

def summary_table():
    print("=" * 75)
    print("SUMMARY: Total Error Budget")
    print("=" * 75)
    print()

    header = (f"  {'Benchmark':>22}  {'v(km/s)':>7}  {'sigma/m':>10}  "
              f"{'dN_steps':>9}  {'dl_max':>8}  {'dx_min':>8}  "
              f"{'vs scipy':>9}  {'Total':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for bm in BENCHMARKS:
        for v in VELOCITIES:
            sig_def = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v)

            # Integrator: default vs N=16000
            sig_16k = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v,
                                   N_steps_override=16000)
            err_N = abs(sig_def - sig_16k) / sig_def * 100 if sig_def > 0 else 0

            # Truncation: l_max vs l_max+20
            kappa, lam, l_max = get_params(bm, v)
            sig_lp = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v,
                                  l_max_override=l_max + 20)
            err_l = abs(sig_def - sig_lp) / sig_def * 100 if sig_def > 0 else 0

            # Prescription: max deviation among 0.8, 1.2
            sig_08 = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v,
                                  x_min_factor=0.8)
            sig_12 = sigma_T_flex(bm["m_chi"], bm["m_phi"], bm["alpha"], v,
                                  x_min_factor=1.2)
            err_x = max(abs(sig_def - sig_08), abs(sig_def - sig_12))
            err_x = err_x / sig_def * 100 if sig_def > 0 else 0

            # vs scipy ground truth (same l_max)
            sig_scipy = scipy_sigma_T(bm["m_chi"], bm["m_phi"], bm["alpha"], v)
            err_scipy = abs(sig_def - sig_scipy) / sig_scipy * 100 if sig_scipy > 0 else 0

            # Total: quadrature sum
            total = math.sqrt(err_N**2 + err_l**2 + err_x**2)

            name_short = bm["name"].split("(")[1].rstrip(")") if "(" in bm["name"] else bm["name"]
            print(f"  {name_short:>22}  {v:7.0f}  {sig_def:10.4f}  "
                  f"{err_N:8.4f}%  {err_l:7.4f}%  {err_x:7.4f}%  "
                  f"{err_scipy:8.4f}%  {total:7.4f}%")

    print()
    print("  dN_steps = |sigma(default) - sigma(16000)| / sigma(default)")
    print("  dl_max   = |sigma(l_max) - sigma(l_max+20)| / sigma(l_max)")
    print("  dx_min   = max(|sigma(1.0) - sigma(0.8)|, |sigma(1.0) - sigma(1.2)|) / sigma(1.0)")
    print("  vs scipy = |Numba RK4 - scipy RK45(rtol=1e-10)| / scipy  (same l_max)")
    print("  Total    = sqrt(dN^2 + dl^2 + dx^2)")
    print()

# ==============================================================
#  Main
# ==============================================================

def main():
    print("=" * 75)
    print("V10 - error_budget.py")
    print("Systematic Numerical Error Analysis for VPM Solver")
    print("  Benchmarks: BP1, MAP, BP17 (from config.json + relic CSV)")
    print("=" * 75)
    t0 = time.time()

    # JIT warmup
    print("\n  Warming up Numba JIT...", flush=True)
    _ = sigma_T_flex(10.0, 1e-3, 1e-4, 30.0)
    print("  JIT ready.\n")

    test_integrator()
    test_truncation()
    test_prescription()
    summary_table()

    elapsed = time.time() - t0
    print(f"  Total runtime: {elapsed:.1f}s")

if __name__ == "__main__":
    main()


if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
        from tg_notify import notify
        notify("\u2705 error_budget done!")
    except Exception:
        pass
