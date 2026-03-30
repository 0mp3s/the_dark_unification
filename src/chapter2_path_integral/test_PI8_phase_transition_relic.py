#!/usr/bin/env python3
r"""
Test PI-8: First-Order Phase Transition in φ  →  Entropy Dilution  →  Relic Fix
================================================================================
Core question:
  Does V_eff(φ, T) undergo a FIRST-ORDER phase transition at some T_c?
  If yes  → latent heat L → entropy dilution D = 1 + L/(T_c s_before)
          → Y_final = Y_freeze-out / D → corrected Ωh²
  If no   → phase transition ruled out; need alternative mechanism.

Physics:
  V_eff(φ, T) = V_tree(φ) + V_CW(φ) + V_T(φ, T)

  V_tree = −½ m_φ² φ² + (λ/4) φ⁴
    where v_φ² = m_φ²/λ  →  λ = m_φ²/v_φ²
    and   v_φ = m_φ / √(2λ)  (symmetry-broken vacuum)

  V_CW from Majorana χ loop (n_f = 2 real dof):
    V_CW(φ) = −2/(64π²) M_χ⁴(φ) [ln(M_χ²(φ)/μ²) − 3/2]
    M_χ(φ) = y φ / √2  (Yukawa mass)

  V_T  from finite-T corrections:
    Boson  (φ):  T⁴/(2π²) × J_B(m_φ²(φ)/T²)
    Fermion (χ): −g_χ × T⁴/(2π²) × J_F(M_χ²(φ)/T²)

  First-order PT signature:
    At high T: V_eff has minimum at φ=0 (symmetric phase)
    At low T:  V_eff has minimum at φ=v_φ (broken phase)
    At T_c:    both minima are degenerate — barrier between them.

Config-driven: all parameters from GC (global) + data/config.json (local).
NO hardcoded parameter values.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize

# ── path setup (same pattern as lagrangian_path_integral.py) ──────────
_HERE      = Path(__file__).resolve().parent
_SIDM_ROOT = _HERE.parent
sys.path.insert(0, str(_SIDM_ROOT))
sys.path.insert(0, str(_SIDM_ROOT / "core"))

from core.global_config import GC

# ── physical constants from GC ────────────────────────────────────────
_PC         = GC.physical_constants()
GEV2_TO_CM2 = _PC["GEV2_to_cm2"]
GEV_IN_G    = _PC["GeV_in_g"]
M_PL_GEV    = _PC["m_pl_GeV"]

_CC         = GC.cosmological_constants()
OMEGA_TARGET = _CC["omega_h2_target"]
G_CHI       = _CC["g_chi_Majorana"]
G_STAR_S    = _CC["g_star_s_20_90_GeV"]    # g*_s at T ~ 10–100 GeV

# ── local config ──────────────────────────────────────────────────────
_LOCAL_CFG_PATH = _HERE / "data" / "config.json"
with open(_LOCAL_CFG_PATH, "r", encoding="utf-8") as _fh:
    _CFG = json.load(_fh)

N_F_MAJORANA = _CFG["renormalization"]["n_f_majorana"]        # 2
K_A4         = _CFG["renormalization"]["A4_mass_formula"]["K_A4"]  # 9
PLANCK_SV    = _CFG["missing_from_GC"]["planck_sv_cm3_s"]     # 3e-26 cm³/s

# ── benchmark loader ──────────────────────────────────────────────────
def _load_benchmarks():
    """Load benchmark points from GC, driven by config test_benchmarks."""
    bps = []
    for label in _CFG["test_benchmarks"]:
        bp = GC.benchmark(label)
        m_chi = bp["m_chi_GeV"]
        m_phi = bp["m_phi_MeV"] * 1e-3       # → GeV
        alpha = bp["alpha"]
        y     = math.sqrt(4.0 * math.pi * alpha)
        bps.append({
            "label":  label,
            "m_chi":  m_chi,
            "m_phi":  m_phi,
            "alpha":  alpha,
            "y":      y,
            "lam":    alpha * m_chi / m_phi,  # λ = αm_χ/m_φ
        })
    return bps

# ══════════════════════════════════════════════════════════════════════
# THERMAL INTEGRALS  J_B, J_F
# ══════════════════════════════════════════════════════════════════════
_GAMMA_E = 0.5772156649

def _J_highT_boson(x):
    """High-T expansion J_B(x²), valid for x = m/T < 1.5."""
    x2 = x * x; x3 = x2 * x; x4 = x2 * x2
    ln_term = np.log(np.clip(x2, 1e-100, None)
                     / (4.0 * np.pi**2 * np.exp(-2 * _GAMMA_E))) - 1.5
    return -np.pi**4 / 45. + (np.pi**2 / 12.) * x2 \
           - (np.pi / 6.) * x3 - (x4 / 32.) * ln_term

def _J_highT_fermion(x):
    """High-T expansion J_F(x²), valid for x = m/T < 1.5."""
    x2 = x * x; x4 = x2 * x2
    ln_term = np.log(np.clip(x2, 1e-100, None)
                     / (np.pi**2 * np.exp(-2 * _GAMMA_E))) - 1.5
    return -7. * np.pi**4 / 360. + (np.pi**2 / 24.) * x2 \
           + (x4 / 32.) * ln_term

def _J_boltzmann(x, boson):
    """Boltzmann tail, valid for x > 5."""
    sign = 1.0 if boson else -1.0
    return sign * np.sqrt(np.pi * x / 2.) * x**2 * np.exp(-x)

def _J_numerical(x2, boson):
    """Exact numerical integration of J_{B/F}."""
    def integrand(k):
        E = np.sqrt(k * k + x2)
        if boson:
            return k * k * np.log(max(1.0 - np.exp(-E), 1e-300))
        else:
            return k * k * np.log(1.0 + np.exp(-E))
    kmax = max(20.0, 10.0 * np.sqrt(x2))
    result, _ = integrate.quad(integrand, 0, kmax, limit=80, epsabs=1e-12)
    return result

def J_thermal(x, boson):
    """J_{B/F}(x²) with x = m/T.  Dispatch to best approximation."""
    x = max(x, 0.0)
    if x < 1.5:
        return _J_highT_boson(x) if boson else _J_highT_fermion(x)
    elif x > 5.0:
        return _J_boltzmann(x, boson)
    else:
        return _J_numerical(x * x, boson)


# ══════════════════════════════════════════════════════════════════════
# EFFECTIVE POTENTIAL  V_eff(φ, T)
# ══════════════════════════════════════════════════════════════════════

class PhiPotential:
    """
    V_eff(φ,T) for the radial mediator field φ.

    Convention: φ = 0 is symmetric phase, φ = v_φ is broken phase.
    The tree-level potential is V = −½μ²φ² + ¼λ_4 φ⁴  with μ² = m_φ².
    v_φ = μ/√λ_4  and  λ_4 = y⁴ / (16π² × correction factor from CW).

    For the initial scan we take the simplest renormalisable form:
      λ_4 ≡ m_φ² / v_φ²
      v_φ ≡ m_phi / y      (so that M_χ = y v_φ/√2 = m_phi/√2 ~ right ballpark)
    This is a zeroth-order estimate; the true quartic gets radiative corrections.
    """

    def __init__(self, bp: dict):
        self.m_chi = bp["m_chi"]
        self.m_phi = bp["m_phi"]
        self.alpha = bp["alpha"]
        self.y     = bp["y"]
        self.label = bp["label"]

        # Tree-level parameters
        # v_φ: VEV such that M_χ(v_φ) = y v_φ / √2 ≈ m_chi
        self.v_phi = math.sqrt(2.0) * self.m_chi / self.y
        self.lam4  = self.m_phi**2 / self.v_phi**2    # quartic coupling
        self.mu2   = self.m_phi**2                     # μ² (mass² parameter)

        # Renormalisation scale = m_chi (following config convention)
        self.mu_ren = self.m_chi

        # Derived
        self.T_fo   = self.m_chi / 20.0

    # -- field-dependent masses ----------------------------------------
    def M_chi_sq(self, phi):
        """Field-dependent χ mass squared: M_χ²(φ) = (y²/2) φ²."""
        return 0.5 * self.y**2 * phi**2

    def M_phi_sq(self, phi):
        """Field-dependent φ mass squared (from tree potential curvature):
           M_φ²(φ) = −μ² + 3 λ₄ φ²
        """
        return -self.mu2 + 3.0 * self.lam4 * phi**2

    # -- tree-level potential ------------------------------------------
    def V_tree(self, phi):
        """V_tree = −½ μ² φ² + ¼ λ₄ φ⁴"""
        return -0.5 * self.mu2 * phi**2 + 0.25 * self.lam4 * phi**4

    # -- Coleman-Weinberg (χ loop, Majorana = 2 real dof) ---------------
    def V_CW(self, phi):
        """
        V_CW = −n_f/(64π²) M_χ⁴(φ) [ln(M_χ²(φ)/μ²) − 3/2]
        n_f = 2 for Majorana  (from config: N_F_MAJORANA)
        Negative sign: fermion loop.
        """
        Mq2 = self.M_chi_sq(phi)
        if Mq2 < 1e-60:
            return 0.0
        log_term = np.log(Mq2 / self.mu_ren**2) - 1.5
        return -float(N_F_MAJORANA) / (64.0 * np.pi**2) * Mq2**2 * log_term

    # -- finite-temperature correction ---------------------------------
    def V_thermal(self, phi, T):
        """
        V_T = T⁴/(2π²) [ J_B(M_φ²(φ)/T²)  −  g_χ J_F(M_χ²(φ)/T²) ]
        g_χ = 2 (Majorana)  — from GC cosmological_constants.
        """
        if T < 1e-30:
            return 0.0

        # φ field-dependent masses
        Mphi2 = self.M_phi_sq(phi)
        Mchi2 = self.M_chi_sq(phi)

        # Avoid sqrt of negative (tachyonic region → use |m²|)
        xb = math.sqrt(max(abs(Mphi2), 0.0)) / T
        xf = math.sqrt(max(Mchi2, 0.0)) / T

        jb = J_thermal(xb, boson=True)
        jf = J_thermal(xf, boson=False)

        pref = T**4 / (2.0 * np.pi**2)
        return pref * (jb - float(G_CHI) * jf)

    # -- full effective potential --------------------------------------
    def V_eff(self, phi, T):
        return self.V_tree(phi) + self.V_CW(phi) + self.V_thermal(phi, T)

    def V_eff_array(self, phi_arr, T):
        """Evaluate V_eff on an array of φ values."""
        return np.array([self.V_eff(p, T) for p in phi_arr])


# ══════════════════════════════════════════════════════════════════════
# PHASE TRANSITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def find_local_minima(phi_arr, V_arr):
    """Find local minima in V(φ) by checking sign changes of dV/dφ."""
    dV = np.diff(V_arr)
    minima = []
    for i in range(len(dV) - 1):
        if dV[i] < 0 and dV[i + 1] >= 0:
            # Minimum between phi_arr[i] and phi_arr[i+2]; take i+1
            minima.append((phi_arr[i + 1], V_arr[i + 1]))
    return minima


def scan_phase_transition(pot: PhiPotential, n_T=200, n_phi=500):
    """
    Scan V_eff(φ, T) over temperature range to detect first-order PT.

    Returns dict with results.
    """
    v = pot.v_phi
    phi_arr = np.linspace(0.0, 2.0 * v, n_phi)

    # Temperature range: from ~ m_phi / 10 up to 10 × m_chi
    T_lo = pot.m_phi / 10.0
    T_hi = 10.0 * pot.m_chi
    T_arr = np.geomspace(T_lo, T_hi, n_T)[::-1]  # scan from high T downward

    results = {
        "label": pot.label,
        "v_phi": v,
        "m_chi": pot.m_chi,
        "m_phi": pot.m_phi,
        "alpha": pot.alpha,
        "lam4":  pot.lam4,
        "T_arr": [],
        "n_minima": [],
        "minima_phi": [],
        "minima_V": [],
        "T_c": None,
        "phi_broken_at_Tc": None,
        "delta_V_at_Tc": None,
        "latent_heat": None,
        "dilution_factor": None,
        "first_order": False,
    }

    prev_n_min = None
    for T in T_arr:
        V_vals = pot.V_eff_array(phi_arr, T)
        lm = find_local_minima(phi_arr, V_vals)
        n_min = len(lm)

        results["T_arr"].append(T)
        results["n_minima"].append(n_min)
        results["minima_phi"].append([m[0] for m in lm])
        results["minima_V"].append([m[1] for m in lm])

        # Detect first-order: two minima with small V difference
        if n_min >= 2:
            # Sort by φ: first is near origin (symmetric), second is broken
            lm_sorted = sorted(lm, key=lambda x: x[0])
            phi_sym, V_sym = lm_sorted[0]
            phi_brk, V_brk = lm_sorted[-1]

            delta_V = abs(V_sym - V_brk)

            # Check if minima are degenerate (relative to depth)
            V_depth = max(abs(V_sym), abs(V_brk), 1e-100)
            if delta_V / V_depth < 0.01:  # within 1% → near-critical
                if results["T_c"] is None:
                    results["T_c"] = T
                    results["phi_broken_at_Tc"] = phi_brk
                    results["delta_V_at_Tc"] = delta_V
                    results["first_order"] = True

        prev_n_min = n_min

    return results


def compute_entropy_dilution(T_c, latent_heat_density, g_star_s=None):
    """
    Compute dilution factor D from latent heat injection.

    D = 1 + L / (T_c × s_before)
    s_before = (2π²/45) g*_s T_c³
    L = latent heat density [GeV⁴]
    """
    if g_star_s is None:
        g_star_s = G_STAR_S
    s_before = (2.0 * np.pi**2 / 45.0) * g_star_s * T_c**3
    D = 1.0 + latent_heat_density / (T_c * s_before)
    return D, s_before


def compute_latent_heat(pot: PhiPotential, T_c, phi_broken):
    """
    Latent heat for first-order PT:
      L = T_c × d(ΔV)/dT |_{T_c}  −  ΔV(T_c)
    Estimated numerically.
    """
    dT = T_c * 1e-4
    T_plus  = T_c + dT
    T_minus = T_c - dT

    V_sym_p = pot.V_eff(0.0, T_plus)
    V_brk_p = pot.V_eff(phi_broken, T_plus)
    dV_plus = V_sym_p - V_brk_p

    V_sym_m = pot.V_eff(0.0, T_minus)
    V_brk_m = pot.V_eff(phi_broken, T_minus)
    dV_minus = V_sym_m - V_brk_m

    d_deltaV_dT = (dV_plus - dV_minus) / (2.0 * dT)

    delta_V = pot.V_eff(0.0, T_c) - pot.V_eff(phi_broken, T_c)

    L = T_c * d_deltaV_dT - delta_V
    return L


def s_wave_cross_section(m_chi, alpha):
    """⟨σv⟩_{s-wave} = π α² / (4 m_χ²)  in GeV⁻²  →  cm³/s."""
    sv_gev2 = math.pi * alpha**2 / (4.0 * m_chi**2)
    sv_cm3_s = sv_gev2 * GEV2_TO_CM2 * 3e10  # × c in cm/s
    return sv_cm3_s


def relic_abundance_swave(m_chi, alpha):
    """
    Approximate Ωh² from s-wave freeze-out (Kolb-Turner):
      Ωh² ≈ (3e-26 / ⟨σv⟩) × 0.120
    """
    sv = s_wave_cross_section(m_chi, alpha)
    return PLANCK_SV / sv * OMEGA_TARGET


# ══════════════════════════════════════════════════════════════════════
# MAIN — RUN PI-8 FOR ALL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("TEST PI-8: FIRST-ORDER PHASE TRANSITION → ENTROPY DILUTION → RELIC FIX")
    print("=" * 72)

    benchmarks = _load_benchmarks()

    for bp in benchmarks:
        print(f"\n{'━' * 72}")
        print(f"  BENCHMARK: {bp['label']}")
        print(f"{'━' * 72}")
        print(f"  m_χ     = {bp['m_chi']*1e3:.2f} MeV")
        print(f"  m_φ     = {bp['m_phi']*1e3:.2f} MeV")
        print(f"  α       = {bp['alpha']:.4e}")
        print(f"  y       = {bp['y']:.6f}")
        print(f"  λ       = {bp['lam']:.1f}")

        pot = PhiPotential(bp)
        print(f"  v_φ     = {pot.v_phi:.4e} GeV")
        print(f"  λ₄      = {pot.lam4:.4e}")
        print(f"  T_fo    = {pot.T_fo*1e3:.2f} MeV")

        # ── s-wave relic without correction ───────────────────────────
        sv = s_wave_cross_section(bp["m_chi"], bp["alpha"])
        omega_naive = relic_abundance_swave(bp["m_chi"], bp["alpha"])
        ratio_planck = sv / PLANCK_SV

        print(f"\n  ⟨σv⟩_s-wave = {sv:.3e} cm³/s")
        print(f"  ⟨σv⟩/Planck = {ratio_planck:.3f}")
        print(f"  Ωh² (naive) = {omega_naive:.4f}  (target = {OMEGA_TARGET})")
        print(f"  D needed    = {omega_naive / OMEGA_TARGET:.2f}")

        # ── scan V_eff(φ, T) ──────────────────────────────────────────
        print(f"\n  Scanning V_eff(φ, T) ...")
        res = scan_phase_transition(pot, n_T=300, n_phi=800)

        # ── report structure of V_eff at key temperatures ─────────────
        print(f"\n  {'T [MeV]':>10}  {'T/m_χ':>8}  {'# minima':>9}  "
              f"{'φ_min [GeV]':>24}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*24}")
        for i in range(0, len(res["T_arr"]), max(1, len(res["T_arr"]) // 20)):
            T = res["T_arr"][i]
            nm = res["n_minima"][i]
            phis = res["minima_phi"][i]
            phi_str = ", ".join(f"{p:.3e}" for p in phis)
            print(f"  {T*1e3:>10.2f}  {T/bp['m_chi']:>8.3f}  {nm:>9d}  "
                  f"{phi_str:>24s}")

        # ── phase transition result ───────────────────────────────────
        print(f"\n  {'─'*60}")
        if res["first_order"]:
            T_c = res["T_c"]
            phi_brk = res["phi_broken_at_Tc"]
            print(f"  ✅ FIRST-ORDER PHASE TRANSITION DETECTED")
            print(f"     T_c = {T_c*1e3:.4f} MeV  "
                  f"({T_c/bp['m_chi']:.4f} m_χ)")
            print(f"     φ_broken(T_c) = {phi_brk:.4e} GeV")

            L = compute_latent_heat(pot, T_c, phi_brk)
            print(f"     Latent heat L = {L:.4e} GeV⁴")

            D, s_before = compute_entropy_dilution(T_c, L)
            print(f"     s_before = {s_before:.4e} GeV³")
            print(f"     Dilution factor D = {D:.4f}")

            omega_corrected = omega_naive / D
            print(f"\n     Ωh² (naive)     = {omega_naive:.4f}")
            print(f"     Ωh² (corrected) = {omega_corrected:.4f}")
            print(f"     Ωh² (target)    = {OMEGA_TARGET}")
            print(f"     Match? {'✅ YES' if abs(omega_corrected - OMEGA_TARGET) / OMEGA_TARGET < 0.1 else '✗ NO'}")

            D_needed = omega_naive / OMEGA_TARGET
            print(f"\n     D needed for exact match = {D_needed:.2f}")
            print(f"     D achieved               = {D:.4f}")
        else:
            print(f"  ✗ NO FIRST-ORDER PHASE TRANSITION FOUND")
            print(f"    V_eff(φ,T) has single minimum at all temperatures.")
            print(f"    Entropy dilution via φ phase transition is ruled out")
            print(f"    for this benchmark.")
            print(f"\n    Alternative mechanisms needed:")
            print(f"    - Freeze-in (dark sector starts empty)")
            print(f"    - Asymmetric dark matter")
            print(f"    - 3→2 number-changing processes")

    # ── summary table ─────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print("SUMMARY — PI-8 RESULTS")
    print(f"{'═'*72}")
    print(f"\n  {'BP':>12}  {'⟨σv⟩/Planck':>12}  {'Ωh²(naive)':>11}  "
          f"{'1st-order?':>11}  {'D':>8}  {'Ωh²(corr)':>11}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*11}  {'─'*11}  {'─'*8}  {'─'*11}")

    for bp in benchmarks:
        sv = s_wave_cross_section(bp["m_chi"], bp["alpha"])
        omega_n = relic_abundance_swave(bp["m_chi"], bp["alpha"])
        # Re-scan (fast, same as above)
        pot = PhiPotential(bp)
        res = scan_phase_transition(pot, n_T=300, n_phi=800)

        if res["first_order"]:
            L = compute_latent_heat(pot, res["T_c"], res["phi_broken_at_Tc"])
            D, _ = compute_entropy_dilution(res["T_c"], L)
            omega_c = omega_n / D
            print(f"  {bp['label']:>12}  {sv/PLANCK_SV:>12.3f}  "
                  f"{omega_n:>11.4f}  {'YES':>11}  {D:>8.3f}  {omega_c:>11.4f}")
        else:
            print(f"  {bp['label']:>12}  {sv/PLANCK_SV:>12.3f}  "
                  f"{omega_n:>11.4f}  {'NO':>11}  {'—':>8}  {'—':>11}")

    print(f"\n  Target: Ωh² = {OMEGA_TARGET}")
    print(f"\n{'═'*72}")
    print("PI-8 COMPLETE")
    print(f"{'═'*72}")


if __name__ == "__main__":
    main()
