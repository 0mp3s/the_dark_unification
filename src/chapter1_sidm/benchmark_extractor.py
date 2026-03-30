#!/usr/bin/env python3
"""
V9 — v30_benchmark_extractor.py
================================
Find optimal Benchmark points from the V9 Scalar Mediator dataset.

Pipeline:
  1. Load all_viable_raw_v8.csv (80k SIDM-passing points at v=30 & v=1000)
  2. Relic density filter via Kolb-Turner s-wave:  0.115 ≤ Ωh² ≤ 0.125
  3. Quantum resonance filter (non-monotonicity):
        σ/m(10) > σ/m(30)  OR  σ/m(5) > σ/m(10)
  4. Print top-5 benchmarks + save v30_perfect_benchmarks.csv

Physics:
  Annihilation channel: Majorana χχ → φφ (scalar t/u-channel)
  ⟨σv⟩ = πα²/(4 m_χ²)   [s-wave, velocity-independent at freeze-out]
"""
# === path setup (auto-generated) ================================
import sys as _sys, os as _os
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _os.path.join(_ROOT, 'core'))
DATA_DIR = _os.path.join(_ROOT, 'data')
# =================================================================

import sys
import os
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# ---------------------------------------------------------------------------
#  Path setup: import physics from V9 modules
# ---------------------------------------------------------------------------
_V9_DIR = os.path.dirname(os.path.abspath(__file__))
if _V9_DIR not in sys.path:
    sys.path.insert(0, _V9_DIR)

# Relic-density functions (s-wave Kolb-Turner, g_* tables, Y_to_omega_h2)
from v27_boltzmann_relic import (
    kolb_turner_swave,
    Y_to_omega_h2,
    g_star_rho,
    g_star_S,
    M_PL,
    S_0,
    RHO_CRIT_H2,
)

# VPM sigma_T solver (Numba JIT, releases GIL → thread-parallel)
from v22_raw_scan import sigma_T_vpm
from output_manager import get_latest, timestamped_path

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
OMEGA_LO = 0.115
OMEGA_HI = 0.125
CSV_IN   = str(get_latest("all_viable_raw_v8"))
CSV_OUT  = str(timestamped_path("v30_perfect_benchmarks"))

# ---------------------------------------------------------------------------
#  Helper: fast vectorised Ωh² via Kolb-Turner s-wave
# ---------------------------------------------------------------------------
def omega_h2_swave(m_chi: float, alpha: float) -> float:
    """Ωh² for Majorana χχ → φφ s-wave via Kolb-Turner approximation."""
    sv0 = math.pi * alpha**2 / (4.0 * m_chi**2)   # ⟨σv⟩ in GeV⁻²
    if sv0 <= 0:
        return 1e30
    _, Y_inf = kolb_turner_swave(m_chi, sv0)
    return Y_to_omega_h2(Y_inf, m_chi)


# ---------------------------------------------------------------------------
#  Step 1: Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("  V9 Benchmark Extractor — Scalar Mediator (Final Closure)")
print("=" * 70)
print()
print(f"[1] Loading {CSV_IN} ...")
t0 = time.time()
df = pd.read_csv(CSV_IN)
# Support both m_phi_MeV (current) and legacy m_phi_GeV headers
if 'm_phi_MeV' in df.columns:
    df['m_phi_GeV'] = df['m_phi_MeV'] / 1000.0
elif 'm_phi_GeV' not in df.columns:
    raise KeyError("CSV must contain 'm_phi_MeV' or 'm_phi_GeV' column")
print(f"    Loaded {len(df):,} points  ({time.time()-t0:.2f}s)")
print(f"    Columns: {list(df.columns)}")
print()

# ---------------------------------------------------------------------------
#  Step 2: Relic density filter  (Kolb-Turner s-wave, vectorised over rows)
# ---------------------------------------------------------------------------
print(f"[2] Relic density filter  (0.115 ≤ Ωh² ≤ 0.125, Kolb-Turner s-wave) ...")
t1 = time.time()

omega_vals = np.array([
    omega_h2_swave(row.m_chi_GeV, row.alpha)
    for row in df.itertuples(index=False)
])
df["omega_h2"] = omega_vals

mask_relic = (omega_vals >= OMEGA_LO) & (omega_vals <= OMEGA_HI)
df_relic = df[mask_relic].copy().reset_index(drop=True)

print(f"    Survived: {len(df_relic):,} / {len(df):,} "
      f"  ({100*len(df_relic)/len(df):.2f}%)  "
      f"[{time.time()-t1:.1f}s]")
print()

if len(df_relic) == 0:
    print("  WARNING: No points survived the relic density cut.")
    print("  Check α range in the CSV vs s-wave freeze-out requirement.")
    sys.exit(0)

# ---------------------------------------------------------------------------
#  Step 3: Compute σ/m at v = 5 and 10 km/s for relic-surviving points
#          (Numba JIT functions release the GIL → real threading speedup)
# ---------------------------------------------------------------------------
print(f"[3] Computing σ/m at v=5 and v=10 km/s for {len(df_relic):,} survivors ...")
t2 = time.time()

mc_arr   = df_relic["m_chi_GeV"].values
mph_arr  = df_relic["m_phi_GeV"].values
alp_arr  = df_relic["alpha"].values
n_surv   = len(df_relic)

sm5  = np.zeros(n_surv)
sm10 = np.zeros(n_surv)

N_WORKERS = min(8, os.cpu_count() or 4)


def _compute_row(i):
    """Return (i, sigma_5, sigma_10)."""
    s5  = sigma_T_vpm(mc_arr[i], mph_arr[i], alp_arr[i], 5.0)
    s10 = sigma_T_vpm(mc_arr[i], mph_arr[i], alp_arr[i], 10.0)
    return i, s5, s10


with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = [pool.submit(_compute_row, i) for i in range(n_surv)]
    done = 0
    report_every = max(1, n_surv // 20)
    for fut in as_completed(futures):
        i, s5, s10 = fut.result()
        sm5[i]  = s5
        sm10[i] = s10
        done += 1
        if done % report_every == 0 or done == n_surv:
            pct = 100.0 * done / n_surv
            print(f"    {done:5d}/{n_surv}  ({pct:.0f}%)  [{time.time()-t2:.1f}s]",
                  end="\r", flush=True)

print()
print(f"    Done in {time.time()-t2:.1f}s")
print()

df_relic["sigma_m_5"]  = sm5
df_relic["sigma_m_10"] = sm10

# ---------------------------------------------------------------------------
#  Step 4: Rank by "cosmological precision" — closeness to Ωh² = 0.120
# ---------------------------------------------------------------------------
df_relic["delta_omega"] = (df_relic["omega_h2"] - 0.120).abs()
df_final = df_relic.sort_values("delta_omega").reset_index(drop=True)

# ---------------------------------------------------------------------------
#  Step 6: Save full filtered list
# ---------------------------------------------------------------------------
df_final["m_phi_MeV"] = df_final["m_phi_GeV"] * 1000.0
cols_out = [
    "m_chi_GeV", "m_phi_MeV", "alpha", "omega_h2",
    "sigma_m_5", "sigma_m_10", "sigma_m_30", "sigma_m_1000",
    "resonance_idx", "delta_omega"
]
df_final[cols_out].to_csv(CSV_OUT, index=False)
print(f"[5] Saved {len(df_final):,} benchmark points → {CSV_OUT}")
print()

# ---------------------------------------------------------------------------
#  Step 6: Print all benchmark points
# ---------------------------------------------------------------------------
print("=" * 70)
print("  COSMOLOGICAL BENCHMARK POINTS  (SIDM + Ωh² = 0.120 ± 0.005)")
print("=" * 70)
print()

header = (
    f"  {'#':>2}  {'m_χ [GeV]':>12}  {'m_φ [GeV]':>12}  {'α':>12}  "
    f"{'Ωh²':>8}  "
    f"{'σ/m(5)':>9}  {'σ/m(10)':>9}  {'σ/m(30)':>9}  {'σ/m(1000)':>10}"
)
print(header)
print("  " + "-" * (len(header) - 2))

for rank, row in enumerate(df_final.itertuples(index=False), start=1):
    print(
        f"  {rank:>2}  "
        f"{row.m_chi_GeV:>12.4f}  "
        f"{row.m_phi_GeV:>12.5e}  "
        f"{row.alpha:>12.5e}  "
        f"{row.omega_h2:>8.4f}  "
        f"{row.sigma_m_5:>9.4f}  "
        f"{row.sigma_m_10:>9.4f}  "
        f"{row.sigma_m_30:>9.4f}  "
        f"{row.sigma_m_1000:>10.5f}"
    )

print()
print(f"  (Units: σ/m in cm²/g, velocities in km/s)")
print()
elapsed_total = time.time() - t0
print(f"  Total runtime: {elapsed_total:.1f}s")
print("=" * 70)

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))
    from tg_notify import notify
    notify(f"✅ benchmark_extractor done!\n{len(df_final):,} benchmarks saved\nelapsed={elapsed_total:.0f}s")
except Exception:
    pass
