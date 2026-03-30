#!/usr/bin/env python3
"""
vpm_scan/vpm_born_ratio.py
==========================
VPM / Born cross-section ratio R(v) = sigma_VPM / sigma_Born
for BP1 and MAP benchmarks across SIDM-relevant velocities.

Physics:
  Born approximation is the single-scattering (perturbative) limit,
  valid when lambda << 1. For our benchmarks (lambda ~ 2-49), Born
  underpredicts near resonances and overpredicts in destructive
  interference regions.

  The ratio R(v) = sigma_VPM/sigma_Born quantifies the nonlinear
  (multi-scattering) correction from the full partial-wave solution.

Born sigma_T (Majorana, l-by-l):
  delta_l^Born = kappa * lam * int_0^inf [j_l(kappa*x)]^2 e^{-x} x dx
  sigma_T^Born = (2pi/k^2) * sum_l w_l (2l+1) sin^2(delta_l^Born)

  with w_l = 1 (even l), 3 (odd l) for identical Majorana fermions.

Produces:
  - Console table
  - output/vpm_born_ratio.png
  - output/vpm_born_ratio.pdf
"""
import sys, os, math, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, '..')
sys.path.insert(0, os.path.join(_ROOT, 'core'))

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

from numba import jit
from scipy.integrate import quad
from scipy.special import spherical_jn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load config ──
from config_loader import load_config
from global_config import GC

# ── Constants (sourced from global_config.json) ──
_PC = GC.physical_constants()
GEV2_TO_CM2 = _PC["GEV2_to_cm2"]
GEV_IN_G    = _PC["GeV_in_g"]
C_KM_S      = _PC["c_km_s"]
cfg = load_config(__file__)

_bps = {bp['label']: bp for bp in GC.benchmarks_from_labels(cfg.get('benchmark_labels', []))}
BP1 = _bps['BP1']
MAP = _bps['MAP']

BENCHMARKS = [
    {"label": "BP1",
     "m_chi": BP1['m_chi_GeV'],
     "m_phi": BP1['m_phi_MeV'] * 1e-3,
     "alpha": BP1['alpha']},
    {"label": "MAP",
     "m_chi": MAP['m_chi_GeV'],
     "m_phi": MAP['m_phi_MeV'] * 1e-3,
     "alpha": MAP['alpha']},
]

VELOCITIES = [12, 30, 60, 100, 200, 500, 1000, 2000, 4700]

OUT_DIR = os.path.join(_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Numba VPM solver (production) ──
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
def vpm_phase_shift(l, kappa, lam, x_max, N_steps):
    if lam < 1e-30 or kappa < 1e-30: return 0.0
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
def sigma_T_vpm(m_chi, m_phi, alpha, v_km_s):
    """Full VPM sigma_T/m [cm^2/g] for identical Majorana."""
    v = v_km_s / C_KM_S
    mu = m_chi / 2.0
    k = mu * v
    kappa = k / m_phi
    lam = alpha * m_chi / m_phi
    if kappa < 1e-15: return 0.0

    if kappa < 5:
        x_max, N_steps = 50.0, 4000
    elif kappa < 50:
        x_max, N_steps = 80.0, 8000
    else:
        x_max, N_steps = 100.0, 12000

    l_max = min(max(3, min(int(kappa * x_max), int(kappa) + int(lam) + 20)), 500)

    sigma_sum = 0.0
    for l in range(l_max + 1):
        delta = vpm_phase_shift(l, kappa, lam, x_max, N_steps)
        weight = 1.0 if l % 2 == 0 else 3.0
        contrib = weight * (2*l + 1) * math.sin(delta)**2
        sigma_sum += contrib
        if l > int(kappa) + 1 and sigma_sum > 0:
            if contrib / sigma_sum < 1e-3:
                break

    sigma_GeV2 = 2.0 * math.pi * sigma_sum / (k * k)
    sigma_cm2 = sigma_GeV2 * GEV2_TO_CM2
    return sigma_cm2 / (m_chi * GEV_IN_G)


# ── Born approximation (scipy-based) ──
def born_phase_shift_quad(l, kappa, lam):
    """Born phase shift: delta_l^Born = kappa*lam * int_0^inf [j_l(kappa x)]^2 e^{-x} x dx"""
    def integrand(x):
        z = kappa * x
        jl = spherical_jn(l, z)
        return jl**2 * np.exp(-x) * x
    result, _ = quad(integrand, 0, np.inf, limit=200)
    return kappa * lam * result


def born_s_wave_analytical(kappa, lam):
    """Exact s-wave Born: delta_0 = lam/(4*kappa) * ln(1 + 4*kappa^2)"""
    return lam / (4.0 * kappa) * math.log(1.0 + 4.0 * kappa**2)


def sigma_T_born(m_chi, m_phi, alpha, v_km_s):
    """Born sigma_T/m [cm^2/g] for identical Majorana, summing Born phase shifts."""
    v = v_km_s / C_KM_S
    mu = m_chi / 2.0
    k = mu * v
    kappa = k / m_phi
    lam = alpha * m_chi / m_phi
    if kappa < 1e-15:
        return 0.0

    if kappa < 5:
        x_max = 50.0
    elif kappa < 50:
        x_max = 80.0
    else:
        x_max = 100.0

    l_max = min(max(3, min(int(kappa * x_max), int(kappa) + int(lam) + 20)), 500)

    sigma_sum = 0.0
    for l in range(l_max + 1):
        if l == 0:
            delta_b = born_s_wave_analytical(kappa, lam)
        else:
            delta_b = born_phase_shift_quad(l, kappa, lam)
        weight = 1.0 if l % 2 == 0 else 3.0
        contrib = weight * (2*l + 1) * math.sin(delta_b)**2
        sigma_sum += contrib
        if l > int(kappa) + 1 and sigma_sum > 0:
            if contrib / sigma_sum < 1e-3:
                break

    sigma_GeV2 = 2.0 * math.pi * sigma_sum / (k * k)
    sigma_cm2 = sigma_GeV2 * GEV2_TO_CM2
    return sigma_cm2 / (m_chi * GEV_IN_G)


def main():
    print("=" * 90)
    print("  VPM / Born Cross-Section Ratio R(v) = sigma_VPM / sigma_Born")
    print("  Quantifies nonlinear (multi-scattering) corrections to Born approximation")
    print("=" * 90)

    # Warm up JIT
    sigma_T_vpm(20.0, 10e-3, 1e-3, 100.0)

    results = {}  # results[label][v] = (sig_vpm, sig_born, ratio)

    for bm in BENCHMARKS:
        label = bm['label']
        lam = bm['alpha'] * bm['m_chi'] / bm['m_phi']
        print(f"\n  --- {label}: m_chi={bm['m_chi']:.2f} GeV, "
              f"m_phi={bm['m_phi']*1e3:.2f} MeV, alpha={bm['alpha']:.3e}, "
              f"lambda={lam:.2f} ---")

        results[label] = {}
        print(f"  {'v [km/s]':>10}  {'sig_VPM/m':>12}  {'sig_Born/m':>12}  "
              f"{'R=VPM/Born':>12}  {'kappa':>8}")
        print("  " + "-" * 62)

        for v in VELOCITIES:
            sig_vpm = sigma_T_vpm(bm['m_chi'], bm['m_phi'], bm['alpha'], float(v))

            t_born = time.time()
            sig_born = sigma_T_born(bm['m_chi'], bm['m_phi'], bm['alpha'], float(v))
            dt_born = time.time() - t_born

            v_c = v / C_KM_S
            mu = bm['m_chi'] / 2.0
            kappa = mu * v_c / bm['m_phi']

            ratio = sig_vpm / sig_born if sig_born > 0 else float('inf')
            results[label][v] = (sig_vpm, sig_born, ratio)
            print(f"  {v:>10}  {sig_vpm:>12.6f}  {sig_born:>12.6f}  "
                  f"{ratio:>12.4f}  {kappa:>8.4f}")

    # ── Publication figure ──
    print("\n  Generating VPM/Born ratio plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {'BP1': '#E91E63', 'MAP': '#2196F3'}
    markers = {'BP1': 'o', 'MAP': 's'}

    # Panel 1: sigma_VPM and sigma_Born overlaid
    ax = axes[0]
    for bm in BENCHMARKS:
        label = bm['label']
        vels = sorted(results[label].keys())
        sig_v = [results[label][v][0] for v in vels]
        sig_b = [results[label][v][1] for v in vels]
        ax.plot(vels, sig_v, '-', color=colors[label], lw=2.5,
                marker=markers[label], ms=5, label=f'{label} VPM')
        ax.plot(vels, sig_b, '--', color=colors[label], lw=2,
                marker=markers[label], ms=5, mfc='none', label=f'{label} Born')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$v$ [km/s]', fontsize=13)
    ax.set_ylabel('$\\sigma_T/m$ [cm$^2$/g]', fontsize=13)
    ax.set_title('VPM vs Born cross sections', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Ratio R(v)
    ax = axes[1]
    for bm in BENCHMARKS:
        label = bm['label']
        lam = bm['alpha'] * bm['m_chi'] / bm['m_phi']
        vels = sorted(results[label].keys())
        ratios = [results[label][v][2] for v in vels]
        ax.plot(vels, ratios, '-', color=colors[label], lw=2.5,
                marker=markers[label], ms=6, label=f'{label} ($\\lambda={lam:.1f}$)')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('$v$ [km/s]', fontsize=13)
    ax.set_ylabel('$R(v) = \\sigma_{\\rm VPM}/\\sigma_{\\rm Born}$', fontsize=13)
    ax.set_title('VPM / Born ratio (nonlinear correction)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('Full VPM vs Born Approximation: Secluded Majorana SIDM',
                 fontsize=14, y=1.02)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f'vpm_born_ratio.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: output/vpm_born_ratio.png")
    print(f"  Saved: output/vpm_born_ratio.pdf")

    # ── Summary ──
    print(f"\n{'='*90}")
    print("  SUMMARY")
    print(f"{'='*90}")
    for bm in BENCHMARKS:
        label = bm['label']
        lam = bm['alpha'] * bm['m_chi'] / bm['m_phi']
        ratios = [results[label][v][2] for v in VELOCITIES]
        r_min = min(ratios)
        r_max = max(ratios)
        print(f"  {label} (lam={lam:.1f}): R(v) range [{r_min:.3f}, {r_max:.3f}]")
    print(f"\n  Born fails dramatically near resonances (lambda >> 1).")
    print(f"  At high v, partial waves enter weak-coupling regime => R -> 1.")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
    try:
        from tg_notify import notify
        notify("\u2705 vpm_born_ratio done!")
    except Exception:
        pass
