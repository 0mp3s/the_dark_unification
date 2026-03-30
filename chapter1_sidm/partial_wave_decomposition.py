#!/usr/bin/env python3
"""
vpm_scan/partial_wave_decomposition.py
======================================
Partial-wave decomposition of sigma_T for BP1 and MAP benchmarks.

Shows per-ell contributions sigma_l/m to the total cross section at
each SIDM velocity, revealing which partial waves dominate in each
kinematic regime.

Key physics:
  - Low v (kappa << 1):  s-wave dominated, l=0 carries ~100%
  - Mid v (kappa ~ 1-5): first few partial waves compete; Majorana
    spin-statistics (w=1 even, w=3 odd) create oscillating pattern
  - High v (kappa >> 1):  many partial waves contribute up to l ~ kappa

sigma_l = w_l * (2l+1) * sin^2(delta_l) * 2pi / k^2
sigma_T = sum_l sigma_l

Produces:
  - Console table (per-ell contributions)
  - output/partial_wave_decomposition.png
  - output/partial_wave_decomposition.pdf
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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


# ── Numba VPM solver ──
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


def compute_partial_waves(m_chi, m_phi, alpha, v_km_s, l_max_override=None):
    """Return per-ell data: list of (l, delta_l, sigma_l_over_m, weight, frac)."""
    v = v_km_s / C_KM_S
    mu = m_chi / 2.0
    k = mu * v
    kappa = k / m_phi
    lam = alpha * m_chi / m_phi
    if kappa < 1e-15:
        return [], kappa

    if kappa < 5:
        x_max, N_steps = 50.0, 4000
    elif kappa < 50:
        x_max, N_steps = 80.0, 8000
    else:
        x_max, N_steps = 100.0, 12000

    l_max = l_max_override if l_max_override else min(max(3, min(int(kappa * x_max), int(kappa) + int(lam) + 20)), 500)

    data = []
    sigma_sum = 0.0
    for l in range(l_max + 1):
        delta = vpm_phase_shift(l, kappa, lam, x_max, N_steps)
        weight = 1.0 if l % 2 == 0 else 3.0
        contrib_dimless = weight * (2*l + 1) * math.sin(delta)**2
        sigma_sum += contrib_dimless
        # sigma_l in cm^2/g
        sigma_l_GeV2 = 2.0 * math.pi * contrib_dimless / (k * k)
        sigma_l_cm2 = sigma_l_GeV2 * GEV2_TO_CM2
        sigma_l_over_m = sigma_l_cm2 / (m_chi * GEV_IN_G)
        data.append((l, delta, sigma_l_over_m, weight, contrib_dimless))

    # Compute fractions
    total = sum(d[4] for d in data)
    result = []
    for l, delta, sig_m, w, c in data:
        frac = c / total if total > 0 else 0.0
        result.append((l, delta, sig_m, w, frac))

    return result, kappa


def main():
    print("=" * 90)
    print("  Partial-Wave Decomposition of sigma_T")
    print("  w_l = 1 (even l), 3 (odd l) for identical Majorana fermions")
    print("=" * 90)

    # Warm up JIT
    vpm_phase_shift(0, 1.0, 1.0, 50.0, 4000)

    all_results = {}  # all_results[label][v] = [(l, delta, sig_m, w, frac), ...]

    # Selected velocities for detailed table
    detail_vels = [30, 200, 1000, 4700]

    for bm in BENCHMARKS:
        label = bm['label']
        lam = bm['alpha'] * bm['m_chi'] / bm['m_phi']
        print(f"\n{'='*90}")
        print(f"  {label}: m_chi={bm['m_chi']:.2f} GeV, "
              f"m_phi={bm['m_phi']*1e3:.2f} MeV, alpha={bm['alpha']:.3e}, "
              f"lambda={lam:.2f}")
        print(f"{'='*90}")

        all_results[label] = {}

        for v in VELOCITIES:
            pw_data, kappa = compute_partial_waves(
                bm['m_chi'], bm['m_phi'], bm['alpha'], float(v))
            all_results[label][v] = pw_data

            if v in detail_vels:
                sigma_total = sum(d[2] for d in pw_data)
                print(f"\n  v = {v} km/s (kappa = {kappa:.4f}), "
                      f"sigma_T/m = {sigma_total:.6f} cm^2/g")
                print(f"  {'l':>4}  {'w_l':>4}  {'delta_l':>12}  "
                      f"{'sigma_l/m':>12}  {'frac':>8}")
                print("  " + "-" * 48)
                for l, delta, sig_m, w, frac in pw_data:
                    if frac > 1e-4:  # only show significant contributions
                        print(f"  {l:>4}  {int(w):>4}  {delta:>12.6f}  "
                              f"{sig_m:>12.6e}  {frac:>7.2%}")

            # Compact summary for non-detail velocities
            if v not in detail_vels:
                sigma_total = sum(d[2] for d in pw_data)
                n_sig = sum(1 for d in pw_data if d[4] > 0.01)
                l_dom = max(pw_data, key=lambda d: d[4])[0] if pw_data else 0
                print(f"  v={v:>5} km/s: sigma/m={sigma_total:.6f}, "
                      f"kappa={kappa:.4f}, dominant l={l_dom}, "
                      f"{n_sig} waves > 1%")

    # ══════════════════════════════════════════════════════════════
    #  Publication figure: 2x2 grid
    # ══════════════════════════════════════════════════════════════
    print("\n  Generating partial-wave decomposition plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors_l = plt.cm.tab20(np.linspace(0, 1, 20))

    for idx, bm in enumerate(BENCHMARKS):
        label = bm['label']
        lam = bm['alpha'] * bm['m_chi'] / bm['m_phi']

        # Top panel: stacked bar chart at 4 velocities
        ax = axes[0, idx]
        bar_vels = detail_vels
        x_pos = np.arange(len(bar_vels))
        bar_width = 0.6
        max_l = 0
        for v in bar_vels:
            pw = all_results[label][v]
            for l, _, _, _, frac in pw:
                if frac > 0.005:
                    max_l = max(max_l, l)

        bottoms = np.zeros(len(bar_vels))
        for l in range(max_l + 1):
            fracs = []
            for v in bar_vels:
                pw = all_results[label][v]
                if l < len(pw):
                    fracs.append(pw[l][4])
                else:
                    fracs.append(0.0)
            fracs = np.array(fracs)
            if np.max(fracs) > 0.005:
                c = colors_l[l % 20]
                lbl = f'$\\ell={l}$' if l <= 8 else (f'$\\ell={l}$' if l == max_l else None)
                ax.bar(x_pos, fracs, bar_width, bottom=bottoms, color=c,
                       label=lbl, alpha=0.8, edgecolor='white', linewidth=0.5)
                bottoms += fracs

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{v}' for v in bar_vels])
        ax.set_xlabel('$v$ [km/s]', fontsize=12)
        ax.set_ylabel('Fraction of $\\sigma_T$', fontsize=12)
        ax.set_title(f'{label} ($\\lambda={lam:.1f}$): Partial-wave fractions',
                     fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, ncol=3, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')

        # Bottom panel: heatmap of |delta_l| vs (v, l)
        ax = axes[1, idx]
        l_range = list(range(max_l + 1))
        heatmap = np.zeros((len(VELOCITIES), len(l_range)))
        for vi, v in enumerate(VELOCITIES):
            pw = all_results[label][v]
            for li, l in enumerate(l_range):
                if l < len(pw):
                    heatmap[vi, li] = abs(pw[l][1])  # |delta_l|

        heatmap[heatmap < 1e-10] = 1e-10  # avoid log(0)
        im = ax.imshow(heatmap.T, aspect='auto', origin='lower',
                       norm=LogNorm(vmin=1e-6, vmax=max(3.14, heatmap.max())),
                       cmap='viridis',
                       extent=[-0.5, len(VELOCITIES)-0.5, -0.5, len(l_range)-0.5])
        ax.set_xticks(range(len(VELOCITIES)))
        ax.set_xticklabels([str(v) for v in VELOCITIES], fontsize=8, rotation=45)
        ax.set_xlabel('$v$ [km/s]', fontsize=12)
        ax.set_ylabel('Partial wave $\\ell$', fontsize=12)
        ax.set_title(f'{label}: Phase shifts $|\\delta_\\ell|$', fontsize=12)

        # Mark unitarity limit delta = pi/2
        for vi in range(len(VELOCITIES)):
            for li in range(len(l_range)):
                if heatmap[vi, li] > math.pi/2:
                    ax.plot(vi, li, 'rx', ms=6, mew=1.5)

        cb = plt.colorbar(im, ax=ax, label='$|\\delta_\\ell|$ [rad]')

    fig.suptitle('Partial-Wave Decomposition: Secluded Majorana SIDM',
                 fontsize=14, y=1.01)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f'partial_wave_decomposition.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: output/partial_wave_decomposition.png")
    print(f"  Saved: output/partial_wave_decomposition.pdf")

    # ── Even/Odd summary ──
    print(f"\n{'='*90}")
    print("  EVEN vs ODD PARTIAL WAVE CONTRIBUTIONS (Majorana quantum statistics)")
    print(f"{'='*90}")
    for bm in BENCHMARKS:
        label = bm['label']
        print(f"\n  {label}:")
        print(f"  {'v [km/s]':>10}  {'even frac':>10}  {'odd frac':>10}  {'odd/even':>10}")
        print("  " + "-" * 46)
        for v in VELOCITIES:
            pw = all_results[label][v]
            even_sum = sum(d[2] for d in pw if d[0] % 2 == 0)
            odd_sum = sum(d[2] for d in pw if d[0] % 2 == 1)
            total = even_sum + odd_sum
            ef = even_sum / total if total > 0 else 0
            of_ = odd_sum / total if total > 0 else 0
            ratio = odd_sum / even_sum if even_sum > 0 else float('inf')
            print(f"  {v:>10}  {ef:>9.2%}  {of_:>9.2%}  {ratio:>10.3f}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
    try:
        from tg_notify import notify
        notify("\u2705 partial_wave_decomposition done!")
    except Exception:
        pass
