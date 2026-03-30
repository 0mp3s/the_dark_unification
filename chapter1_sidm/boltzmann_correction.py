#!/usr/bin/env python3
"""
V9 — v30_boltzmann_correction.py  (temporary)
================================================
Verify BP1 with the EXACT numerical Boltzmann solver (not Kolb-Turner).

Steps:
  1. Take BP1 parameters: m_χ = 13.895 GeV, m_φ = 7.34e-3 GeV
  2. Use solve_boltzmann from v27 (RK4 numerical)
  3. Bisect α to find Ωh² = 0.120 exactly
  4. Compute σ/m at v=30,1000 km/s via sigma_T_vpm from v22
"""
# === path setup (auto-generated) ================================
import sys as _sys, os as _os
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _os.path.join(_ROOT, 'core'))
DATA_DIR = _os.path.join(_ROOT, 'data')
# =================================================================

import sys, math, time, io
import numpy as np
from output_manager import timestamped_path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Tee: write to both console and timestamped file
_OUT_PATH = str(timestamped_path("v30_boltzmann_correction", ext=".txt"))
_out_file = open(_OUT_PATH, 'w', encoding='utf-8')

class _Tee:
    def __init__(self, *streams): self._s = streams
    def write(self, msg):
        for s in self._s: s.write(msg)
    def flush(self):
        for s in self._s: s.flush()
    @property
    def encoding(self): return self._s[0].encoding

sys.stdout = _Tee(sys.stdout, _out_file)

# --- import numerical Boltzmann solver from v27 ---
from v27_boltzmann_relic import solve_boltzmann, Y_to_omega_h2, kolb_turner_swave

# --- import VPM cross-section from v22 ---
from v22_raw_scan import sigma_T_vpm

from config_loader import load_config
from run_logger import log_run as _log_run

# ==============================================================
#  BP2 parameters (from config / v30_benchmark_extractor)
# ==============================================================
_CFG = load_config(__file__)
def _get_bp(label):
    for bp in _CFG.get("benchmark_points", []):
        if bp["label"] == label:
            return bp
    return {}
_BP2 = _get_bp("BP2_KT")
M_CHI    = _BP2.get("m_chi_GeV", 18.421)
M_PHI    = _BP2.get("m_phi_MeV", 8.20) * 1e-3   # → GeV
ALPHA_KT = _BP2.get("alpha_KT", 8.19e-4)

TARGET_OMEGA = 0.120

def omega_from_alpha(alpha):
    """Compute Ωh² from α using full numerical Boltzmann."""
    sv0 = math.pi * alpha**2 / (4.0 * M_CHI**2)
    _, Y_arr = solve_boltzmann(M_CHI, sv0)
    Y_inf = Y_arr[-1]
    return Y_to_omega_h2(Y_inf, M_CHI)

# ==============================================================
print("=" * 70)
print("  V9 — BP2 Boltzmann Correction")
print("  m_χ = {:.3f} GeV,  m_φ = {:.2f} MeV".format(M_CHI, M_PHI * 1e3))
print("=" * 70)

# --- Step 1: show KT vs Boltzmann at the original α ---
t0 = time.time()
print("\n[1] Original BP1 (Kolb-Turner α = {:.4e}):".format(ALPHA_KT))
omega_num = omega_from_alpha(ALPHA_KT)
sv0_kt = math.pi * ALPHA_KT**2 / (4.0 * M_CHI**2)
_, Y_kt = kolb_turner_swave(M_CHI, sv0_kt)
omega_kt = Y_to_omega_h2(Y_kt, M_CHI)
print("    Ωh² (Kolb-Turner) = {:.4f}".format(omega_kt))
print("    Ωh² (Numerical)   = {:.4f}".format(omega_num))
print("    Δ = {:.1f}%".format(100 * (omega_num - omega_kt) / omega_kt))

# --- Step 2: bisect α to find Ωh² = 0.120 numerically ---
print("\n[2] Bisecting α for Ωh² = {:.3f} (numerical Boltzmann)...".format(TARGET_OMEGA))

# Ωh² decreases as α increases (more annihilation → less relic)
# Bracket: lower α → higher Ωh², upper α → lower Ωh²
a_lo, a_hi = 4.0e-4, 1.5e-3
omega_lo = omega_from_alpha(a_lo)
omega_hi = omega_from_alpha(a_hi)
print("    α_lo = {:.2e}  → Ωh² = {:.4f}".format(a_lo, omega_lo))
print("    α_hi = {:.2e}  → Ωh² = {:.4f}".format(a_hi, omega_hi))

if not (omega_lo > TARGET_OMEGA > omega_hi):
    print("    ❌ Bracket does not straddle target! Adjusting...")
    # Widen if needed
    while omega_lo < TARGET_OMEGA:
        a_lo /= 2.0
        omega_lo = omega_from_alpha(a_lo)
    while omega_hi > TARGET_OMEGA:
        a_hi *= 2.0
        omega_hi = omega_from_alpha(a_hi)
    print("    Adjusted: α ∈ [{:.2e}, {:.2e}]".format(a_lo, a_hi))

for iteration in range(60):
    a_mid = math.sqrt(a_lo * a_hi)  # geometric mean for log-spaced
    omega_mid = omega_from_alpha(a_mid)
    if omega_mid > TARGET_OMEGA:
        a_lo = a_mid
    else:
        a_hi = a_mid
    if abs(omega_mid - TARGET_OMEGA) / TARGET_OMEGA < 1e-5:
        break

alpha_num = a_mid
omega_final = omega_mid
print("    → α_numerical = {:.6e}".format(alpha_num))
print("    → Ωh²         = {:.6f}".format(omega_final))
print("    → Iterations   = {}".format(iteration + 1))

# --- Step 3: compute σ/m at v=30 and v=1000 for updated α ---
print("\n[3] σ/m with updated α:")
print("    Warming up Numba JIT...")
_ = sigma_T_vpm(10.0, 5e-3, 5e-4, 30.0)  # JIT warmup

sm_30  = sigma_T_vpm(M_CHI, M_PHI, alpha_num, 30.0)
sm_1000 = sigma_T_vpm(M_CHI, M_PHI, alpha_num, 1000.0)
print("    σ/m(30 km/s)   = {:.4f} cm²/g".format(sm_30))
print("    σ/m(1000 km/s) = {:.4f} cm²/g".format(sm_1000))

# --- also show original KT values for comparison ---
sm_30_old  = sigma_T_vpm(M_CHI, M_PHI, ALPHA_KT, 30.0)
sm_1000_old = sigma_T_vpm(M_CHI, M_PHI, ALPHA_KT, 1000.0)

# --- Step 4: summary comparison ---
print("\n" + "=" * 70)
print("  BP1 COMPARISON: Kolb-Turner vs Numerical Boltzmann")
print("=" * 70)
print("  {:30s}  {:>14s}  {:>14s}".format("Parameter", "KT (old)", "Numerical"))
print("  " + "-" * 60)
print("  {:30s}  {:>14.3f}  {:>14.3f}".format("m_χ [GeV]", M_CHI, M_CHI))
print("  {:30s}  {:>14.2f}  {:>14.2f}".format("m_φ [MeV]", M_PHI*1e3, M_PHI*1e3))
print("  {:30s}  {:>14.4e}  {:>14.4e}".format("α", ALPHA_KT, alpha_num))
print("  {:30s}  {:>14.4f}  {:>14.4f}".format("Ωh²", omega_kt, omega_final))
print("  {:30s}  {:>14.4f}  {:>14.4f}".format("σ/m(30) [cm²/g]", sm_30_old, sm_30))
print("  {:30s}  {:>14.4f}  {:>14.4f}".format("σ/m(1000) [cm²/g]", sm_1000_old, sm_1000))
print("  " + "-" * 60)
delta_alpha = (alpha_num - ALPHA_KT) / ALPHA_KT * 100
print("  Δα = {:.2f}%".format(delta_alpha))
print("  Δσ/m(30) = {:.2f}%".format((sm_30 - sm_30_old) / sm_30_old * 100))

# SIDM viability check
viable = 1.0 <= sm_30 <= 10.0 and sm_1000 < 0.1
print("\n  SIDM viable? {}".format("✅ YES" if viable else "❌ NO"))
print("  Ωh² = {:.4f} (target {:.3f}) → {:.2f}% off".format(
    omega_final, TARGET_OMEGA, abs(omega_final - TARGET_OMEGA) / TARGET_OMEGA * 100))

elapsed = time.time() - t0
print("\n  Total time: {:.1f}s".format(elapsed))
print("=" * 70)
sys.stdout = sys.__stdout__  # restore before close to avoid flush error
_out_file.flush(); _out_file.close()
print(f"  Saved \u2192 {_OUT_PATH}")
_log_run(
    script="relic_density/boltzmann_correction.py",
    stage="1 - Relic Density verify",
    params={"m_chi_GeV": M_CHI, "m_phi_MeV": round(M_PHI * 1e3, 2), "alpha_KT": ALPHA_KT},
    status="OK" if viable else "PARTIAL",
    duration_sec=round(elapsed, 2),
    notes=f"KT\u2192Boltzmann delta_alpha={delta_alpha:.2f}%",
)
