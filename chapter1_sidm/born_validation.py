#!/usr/bin/env python3
"""
V7 - v21_born_validation.py
============================
Born-limit validation of the VPM phase-shift solver.

Strategy
--------
For lam << 1 (Born regime), partial-wave phase shifts have exact
analytical/quadrature expressions. We compare VPM to Born, accounting
for the centrifugal barrier cutoff that the production solver uses.

Key physics:
  - For l=0 (no barrier): VPM should match Born exactly.
  - For l>0: VPM starts at x_barrier = l/kappa, intentionally skipping the
    classically forbidden region. This gives delta_VPM < delta_Born.
    The ratio delta_VPM/delta_Born approaches 1 as kappa >> l.

Born phase shift (Yukawa, attractive):
  delta_l^Born = kappa * lam * integral_0^inf [j_l(kappa*x)]^2 e^{-x} x dx

Analytical s-wave:
  delta_0^Born = lam/(4*kappa) * ln(1 + 4*kappa^2)

Tests
-----
A. s-wave accuracy: VPM (fine x_min) vs analytical Born
B. Beyond-Born scaling: |delta_VPM - delta_Born| proportional to lam^2
C. Barrier physics: delta_VPM < delta_Born for l>0, ratio -> 1 as kappa -> inf
D. Dirac sigma from l=0 only: matches Born closed form
E. Physical benchmark: V7 working point matches v19
F. Majorana/Dirac ratio consistency
"""

import sys, os, math, time
import numpy as np
from numba import jit
from scipy.integrate import quad
from scipy.special import spherical_jn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from run_logger import RunLogger
from global_config import GC

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── Constants (sourced from global_config.json) ──
_PC = GC.physical_constants()
GEV2_TO_CM2 = _PC["GEV2_to_cm2"]
GEV_IN_G    = _PC["GeV_in_g"]
C_KM_S      = _PC["c_km_s"]

# ==============================================================
#  VPM solver (identical to v18_vpm_scan.py)
# ==============================================================

@jit(nopython=True, cache=True)
def sph_jn_numba(l, z):
    if z < 1e-30:
        return 1.0 if l == 0 else 0.0
    j0 = math.sin(z) / z
    if l == 0:
        return j0
    j1 = math.sin(z) / (z * z) - math.cos(z) / z
    if l == 1:
        return j1
    j_prev, j_curr = j0, j1
    for n in range(1, l):
        j_next = (2 * n + 1) / z * j_curr - j_prev
        j_prev = j_curr
        j_curr = j_next
        if abs(j_curr) < 1e-300:
            return 0.0
    return j_curr


@jit(nopython=True, cache=True)
def sph_yn_numba(l, z):
    if z < 1e-30:
        return -1e300
    y0 = -math.cos(z) / z
    if l == 0:
        return y0
    y1 = -math.cos(z) / (z * z) - math.sin(z) / z
    if l == 1:
        return y1
    y_prev, y_curr = y0, y1
    for n in range(1, l):
        y_next = (2 * n + 1) / z * y_curr - y_prev
        y_prev = y_curr
        y_curr = y_next
        if abs(y_curr) > 1e200:
            return y_curr
    return y_curr


@jit(nopython=True, cache=True)
def _vpm_rhs(l, kappa, lam, x, delta):
    if x < 1e-20:
        return 0.0
    z = kappa * x
    if z < 1e-20:
        return 0.0
    jl = sph_jn_numba(l, z)
    nl = sph_yn_numba(l, z)
    j_hat = z * jl
    n_hat = -z * nl
    cd = math.cos(delta)
    sd = math.sin(delta)
    bracket = j_hat * cd - n_hat * sd
    if not math.isfinite(bracket):
        return 0.0
    pot = lam * math.exp(-x) / (kappa * x)
    val = pot * bracket * bracket
    return val if math.isfinite(val) else 0.0


@jit(nopython=True, cache=True)
def vpm_phase_shift(l, kappa, lam, x_max=50.0, N_steps=4000):
    """Production VPM solver (identical to v18)."""
    if lam < 1e-30 or kappa < 1e-30:
        return 0.0
    x_min = max(1e-5, 0.05 / (kappa + 0.01))
    if l > 0:
        x_barrier = l / kappa
        if x_barrier > x_min:
            x_min = x_barrier
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


@jit(nopython=True, cache=True)
def vpm_phase_shift_fine(l, kappa, lam, x_max=80.0, N_steps=32000):
    """High-resolution VPM for Born validation (l=0 only).
    Uses x_min = 1e-5 (no barrier heuristic) and more steps.
    Only safe for l=0 where n_hat_l does not diverge near z=0.
    """
    if lam < 1e-30 or kappa < 1e-30:
        return 0.0
    x_min = 1e-5
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
#  Born phase shifts (reference benchmarks)
# ==============================================================

def born_phase_shift_quad(l, kappa, lam):
    """Born phase shift via scipy quadrature.
    delta_l^Born = kappa * lam * integral_0^inf [j_l(kappa*x)]^2 e^{-x} x dx
    """
    def integrand(x):
        z = kappa * x
        jl = spherical_jn(l, z)
        return jl**2 * np.exp(-x) * x
    result, _ = quad(integrand, 0, np.inf, limit=200)
    return kappa * lam * result


def born_s_wave_analytical(kappa, lam):
    """Exact analytical s-wave Born:
    delta_0 = lam/(4*kappa) * ln(1 + 4*kappa^2)
    """
    return lam / (4.0 * kappa) * math.log(1.0 + 4.0 * kappa**2)


def born_sigma_dirac_analytical(kappa, lam):
    """Analytical Dirac Born cross section (dimensionless).
    sigma_tilde = 4*pi*lam^2 / (4*kappa^2 + 1)
    """
    return 4.0 * math.pi * lam**2 / (4.0 * kappa**2 + 1.0)


# ==============================================================
#  Test A: s-wave accuracy (fine VPM vs analytical Born)
# ==============================================================

def test_A():
    """s-wave with fine x_min -- tests ODE integrator in isolation."""
    lam = 0.01
    kappas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    print("=" * 90)
    print("TEST A: s-wave VPM (fine x_min=1e-5, N=32000) vs Born analytical (lam=0.01)")
    print("=" * 90)
    print(f"{'kappa':>8} {'d0_VPM_fine':>15} {'d0_Born(A)':>15} {'d0_Born(Q)':>15} {'VPM/A err':>12} {'OK':>4}")
    print("-" * 75)

    all_pass = True
    for kappa in kappas:
        d_vpm = vpm_phase_shift_fine(0, kappa, lam)
        d_anal = born_s_wave_analytical(kappa, lam)
        d_quad = born_phase_shift_quad(0, kappa, lam)

        err = abs(d_vpm - d_anal) / abs(d_anal)
        ok = err < 0.005  # 0.5% threshold
        if not ok:
            all_pass = False
        tag = "v" if ok else "X"
        print(f"{kappa:8.1f} {d_vpm:15.8e} {d_anal:15.8e} {d_quad:15.8e} {err:12.4e} {tag:>4}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\nTest A: {status}")
    return all_pass


# ==============================================================
#  Test B: Beyond-Born scaling  |delta - delta_Born| / lam^2
# ==============================================================

def test_B():
    """At fixed kappa=2, vary lam. (delta_VPM - delta_Born) should scale as lam^2."""
    kappa = 2.0
    lambdas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    print("\n" + "=" * 90)
    print("TEST B: Beyond-Born scaling (kappa=2, l=0, fine VPM)")
    print("  (delta_VPM - delta_Born) / lam^2 -> constant (2nd Born correction)")
    print("=" * 90)
    print(f"{'lam':>10} {'d0_VPM':>15} {'d0_Born':>15} {'diff':>15} {'diff/lam^2':>15}")
    print("-" * 75)

    ratios = []
    for lam in lambdas:
        d_vpm = vpm_phase_shift_fine(0, kappa, lam)
        d_born = born_s_wave_analytical(kappa, lam)
        diff = d_vpm - d_born
        ratio = diff / lam**2
        ratios.append(ratio)
        print(f"{lam:10.4f} {d_vpm:15.8e} {d_born:15.8e} {diff:15.8e} {ratio:15.8e}")

    # Check convergence: last 3 ratios should approach a constant
    last3 = ratios[-3:]
    mean_val = np.mean(last3)
    if abs(mean_val) > 1e-30:
        spread = max(abs(r - mean_val) / abs(mean_val) for r in last3)
    else:
        spread = 0
    converged = spread < 0.30

    print(f"\n  Last 3 diff/lam^2: {last3[0]:.6e}, {last3[1]:.6e}, {last3[2]:.6e}")
    print(f"  Spread: {spread:.1%}  (< 30% required)")
    status = "PASS" if converged else "FAIL"
    print(f"\nTest B: {status}")
    return converged


# ==============================================================
#  Test C: Centrifugal barrier effect on l > 0
# ==============================================================

def test_C():
    """For l > 0, verify delta_VPM < delta_Born (barrier undershoot).
    The ratio delta_VPM/delta_Born should increase toward 1 as kappa grows.
    """
    lam = 0.01
    kappas = [1.0, 3.0, 10.0, 30.0, 50.0]

    print("\n" + "=" * 90)
    print("TEST C: Centrifugal barrier effect (l>0, lam=0.01)")
    print("  Expected: delta_VPM < delta_Born, ratio -> 1 for kappa >> l")
    print("=" * 90)
    print(f"{'kappa':>6} {'l':>3} {'d_VPM':>14} {'d_Born':>14} {'VPM/Born':>10} {'VPM<Born':>8}")
    print("-" * 60)

    all_pass = True
    ratio_by_kappa = {}
    for kappa in kappas:
        N = 4000 if kappa < 5 else (8000 if kappa < 50 else 12000)
        x_max = 50.0 if kappa < 5 else (80.0 if kappa < 50 else 100.0)
        test_l = [1, 2, min(3, int(kappa))]
        for l in test_l:
            d_vpm = vpm_phase_shift(l, kappa, lam, x_max, N)
            d_born = born_phase_shift_quad(l, kappa, lam)
            if abs(d_born) > 1e-20:
                ratio = d_vpm / d_born
            else:
                ratio = 1.0
            undershoot = d_vpm <= d_born * 1.001  # 0.1% tolerance for numerical noise
            if not undershoot:
                all_pass = False
            tag = "v" if undershoot else "X"
            print(f"{kappa:6.1f} {l:3d} {d_vpm:14.7e} {d_born:14.7e} {ratio:10.4f} {tag:>8}")
            if l == 1:
                ratio_by_kappa[kappa] = ratio
        print()

    # Check that ratio increases with kappa for l=1
    ratios_l1 = [ratio_by_kappa[k] for k in sorted(ratio_by_kappa.keys())]
    monotone = all(ratios_l1[i] <= ratios_l1[i+1] + 0.01 for i in range(len(ratios_l1)-1))
    if not monotone:
        all_pass = False
    print(f"  l=1 ratios vs kappa: {['%.4f'%r for r in ratios_l1]}")
    print(f"  Monotone increasing: {'yes' if monotone else 'NO'}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\nTest C: {status}")
    return all_pass


# ==============================================================
#  Test D: s-wave Dirac sigma vs closed form
# ==============================================================

def test_D():
    """For small kappa (s-wave dominated), Dirac sigma from VPM l=0
    should match the full Born analytical result.
    """
    lam = 0.01
    kappas_swave = [0.05, 0.1, 0.2, 0.3]

    print("\n" + "=" * 90)
    print("TEST D: s-wave Dirac sigma_tilde vs analytical Born (small kappa, lam=0.01)")
    print("  sigma_tilde = 4*pi/kappa^2 * sin^2(delta_0)  vs  4*pi*lam^2/(4kappa^2+1)")
    print("=" * 90)
    print(f"{'kappa':>8} {'sig_VPM(l=0)':>15} {'sig_Born(A)':>15} {'Rel.Err':>12} {'OK':>4}")
    print("-" * 60)

    all_pass = True
    for kappa in kappas_swave:
        d0 = vpm_phase_shift_fine(0, kappa, lam)
        sig_vpm_l0 = 4.0 * math.pi / kappa**2 * math.sin(d0)**2
        sig_born = born_sigma_dirac_analytical(kappa, lam)

        err = abs(sig_vpm_l0 - sig_born) / sig_born
        ok = err < 0.10  # 10% tolerance: l>0 Born terms not included in VPM l=0
        if not ok:
            all_pass = False
        tag = "v" if ok else "X"
        print(f"{kappa:8.2f} {sig_vpm_l0:15.6e} {sig_born:15.6e} {err:12.4e} {tag:>4}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\nTest D: {status}")
    return all_pass


# ==============================================================
#  Test E: Physical benchmark
# ==============================================================

def test_E():
    """Production solver at V7 benchmark => sigma/m(30) ~ 4.894."""
    m_chi = 100.0
    m_phi = 2.439998e-3
    alpha = 1.869451e-4
    lam   = alpha * m_chi / m_phi

    v_dwarf = 30.0
    v = v_dwarf / C_KM_S
    mu = m_chi / 2.0
    k = mu * v
    kappa = k / m_phi

    print("\n" + "=" * 90)
    print("TEST E: Physical benchmark -- V7 working point")
    print(f"  m_chi={m_chi} GeV, m_phi={m_phi*1e3:.3f} MeV, alpha={alpha:.4e}")
    print(f"  lam={lam:.2f}, kappa={kappa:.4f}")
    print("=" * 90)

    N = 4000 if kappa < 5 else (8000 if kappa < 50 else 12000)
    x_max = 50.0 if kappa < 5 else (80.0 if kappa < 50 else 100.0)
    l_max = min(max(3, min(int(kappa * x_max), int(kappa) + int(lam) + 20)), 500)

    sigma_sum = 0.0
    for l in range(l_max + 1):
        d = vpm_phase_shift(l, kappa, lam, x_max, N)
        w = 1.0 if l % 2 == 0 else 3.0
        sigma_sum += w * (2*l + 1) * math.sin(d)**2

    sigma_GeV2 = 2.0 * math.pi * sigma_sum / (k * k)
    sigma_cm2 = sigma_GeV2 * GEV2_TO_CM2
    sigma_over_m = sigma_cm2 / (m_chi * GEV_IN_G)

    expected = 4.894
    err = abs(sigma_over_m - expected) / expected

    print(f"  sigma/m(30 km/s) = {sigma_over_m:.3f} cm^2/g")
    print(f"  Expected (v19):    {expected:.3f} cm^2/g")
    print(f"  Relative error:    {err:.2e}")

    ok = err < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"\nTest E: {status}")
    return ok


# ==============================================================
#  Test F: Majorana/Dirac ratio consistency
# ==============================================================

def test_F():
    """The ratio sigma_Majorana / sigma_Dirac is consistent between
    VPM and Born, because the spin-statistics weighting is applied
    to the same set of phase shifts.
    """
    lam = 0.05
    kappas = [1.0, 3.0, 10.0]

    print("\n" + "=" * 90)
    print("TEST F: Majorana/Dirac ratio consistency (lam=0.05)")
    print("  R_VPM = sig_Maj_VPM / sig_Dirac_VPM")
    print("  R_Born = sig_Maj_Born / sig_Dirac_Born")
    print("=" * 90)
    print(f"{'kappa':>8} {'R_VPM':>12} {'R_Born':>12} {'|diff|':>12} {'OK':>4}")
    print("-" * 45)

    all_pass = True
    for kappa in kappas:
        N = 4000 if kappa < 5 else (8000 if kappa < 50 else 12000)
        x_max = 50.0 if kappa < 5 else (80.0 if kappa < 50 else 100.0)
        l_max = min(max(3, min(int(kappa * x_max), int(kappa) + int(lam) + 20)), 500)

        deltas_vpm = [vpm_phase_shift(l, kappa, lam, x_max, N) for l in range(l_max + 1)]
        deltas_born = [born_phase_shift_quad(l, kappa, lam) for l in range(l_max + 1)]

        sig_d_vpm = sum((2*l+1)*math.sin(d)**2 for l,d in enumerate(deltas_vpm))
        sig_d_born = sum((2*l+1)*math.sin(d)**2 for l,d in enumerate(deltas_born))
        sig_m_vpm = sum((1.0 if l%2==0 else 3.0)*(2*l+1)*math.sin(d)**2 for l,d in enumerate(deltas_vpm))
        sig_m_born = sum((1.0 if l%2==0 else 3.0)*(2*l+1)*math.sin(d)**2 for l,d in enumerate(deltas_born))

        R_vpm = sig_m_vpm / sig_d_vpm if sig_d_vpm > 0 else 0
        R_born = sig_m_born / sig_d_born if sig_d_born > 0 else 0

        diff = abs(R_vpm - R_born)
        ok = diff < 0.15
        if not ok:
            all_pass = False
        tag = "v" if ok else "X"
        print(f"{kappa:8.1f} {R_vpm:12.4f} {R_born:12.4f} {diff:12.4e} {tag:>4}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\nTest F: {status}")
    return all_pass


# ==============================================================
#  Main
# ==============================================================

if __name__ == "__main__":
    _rl = RunLogger(
        script="vpm_scan/born_validation.py",
        stage="0 - VPM Validation",
        params={"tests": "A,B,C,D,E,F"},
    )
    _rl.__enter__()
    print("V7 Born-limit validation of VPM solver")
    print("=" * 90)
    t0 = time.time()

    # Warm up Numba JIT
    _ = vpm_phase_shift(0, 1.0, 0.01, 50.0, 1000)
    _ = vpm_phase_shift_fine(0, 1.0, 0.01, 50.0, 1000)

    results = {}
    results['A'] = test_A()
    results['B'] = test_B()
    results['C'] = test_C()
    results['D'] = test_D()
    results['E'] = test_E()
    results['F'] = test_F()

    elapsed = time.time() - t0

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    labels = {
        'A': 's-wave VPM vs Born analytical',
        'B': 'Beyond-Born lam^2 scaling',
        'C': 'Centrifugal barrier (l>0)',
        'D': 's-wave Dirac sigma',
        'E': 'Physical benchmark',
        'F': 'Majorana/Dirac ratio',
    }
    for tid, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  Test {tid} ({labels[tid]}): {status}")

    all_ok = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print(f"  Runtime: {elapsed:.1f} s")
    _rl.set_notes("ALL PASS" if all_ok else "SOME TESTS FAILED")
    _rl.set_status("OK" if all_ok else "PARTIAL")
    _rl.__exit__(None, None, None)


if __name__ == '__main__':
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
        from tg_notify import notify
        notify("\u2705 born_validation done!")
    except Exception:
        pass
