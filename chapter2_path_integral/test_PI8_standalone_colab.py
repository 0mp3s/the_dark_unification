#!/usr/bin/env python3
"""
PI-8 STANDALONE: First-Order Phase Transition → Entropy Dilution → Relic Fix
=============================================================================
Self-contained version — runs on Google Colab, Replit, Kaggle, or any Python 3.8+
with numpy + scipy.  NO project dependencies.

Parameters hardcoded FROM global_config.json (2026-03-29 snapshot).

To run on Colab:
  1. Go to https://colab.research.google.com
  2. File → New Notebook
  3. Paste this entire file into a cell
  4. Click ▶ Run
"""
import math
import numpy as np
from scipy import integrate

# ══════════════════════════════════════════════════════════════════════
# PARAMETERS — from global_config.json + local config.json
# ══════════════════════════════════════════════════════════════════════
BENCHMARKS = {
    "BP1":       {"m_chi_GeV": 54.556, "m_phi_MeV": 12.975, "alpha": 2.645e-3},
    "BP9":       {"m_chi_GeV": 48.329, "m_phi_MeV":  8.657, "alpha": 2.350e-3},
    "BP16":      {"m_chi_GeV": 14.384, "m_phi_MeV":  5.047, "alpha": 7.555e-4},
    "MAP":       {"m_chi_GeV": 98.19,  "m_phi_MeV":  9.66,  "alpha": 3.274e-3},
    "MAP_relic": {"m_chi_GeV": 98.19,  "m_phi_MeV":  9.66,  "alpha": 3.274e-3},
}

# Which BPs to test
TEST_BPS = ["BP1", "MAP", "MAP_relic"]

# Physical constants (from global_config.json)
GEV2_TO_CM2 = 3.8938e-28
OMEGA_TARGET = 0.1200
G_CHI = 2           # Majorana dof
G_STAR_S = 86.25    # g*_s at T ~ 20–90 GeV
N_F_MAJORANA = 2    # Majorana fermion = 2 real dof
PLANCK_SV = 3.0e-26 # cm³/s — canonical thermal cross section

# ══════════════════════════════════════════════════════════════════════
# THERMAL INTEGRALS  J_B, J_F
# ══════════════════════════════════════════════════════════════════════
_GAMMA_E = 0.5772156649

def _J_highT_boson(x):
    x2 = x * x; x3 = x2 * x; x4 = x2 * x2
    ln_term = np.log(max(x2, 1e-100)
                     / (4.0 * np.pi**2 * np.exp(-2 * _GAMMA_E))) - 1.5
    return -np.pi**4 / 45. + (np.pi**2 / 12.) * x2 \
           - (np.pi / 6.) * x3 - (x4 / 32.) * ln_term

def _J_highT_fermion(x):
    x2 = x * x; x4 = x2 * x2
    ln_term = np.log(max(x2, 1e-100)
                     / (np.pi**2 * np.exp(-2 * _GAMMA_E))) - 1.5
    return -7. * np.pi**4 / 360. + (np.pi**2 / 24.) * x2 \
           + (x4 / 32.) * ln_term

def _J_boltzmann(x, boson):
    sign = 1.0 if boson else -1.0
    return sign * np.sqrt(np.pi * x / 2.) * x**2 * np.exp(-x)

def _J_numerical(x2, boson):
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
    def __init__(self, label, m_chi_GeV, m_phi_MeV, alpha):
        self.label = label
        self.m_chi = m_chi_GeV
        self.m_phi = m_phi_MeV * 1e-3   # → GeV
        self.alpha = alpha
        self.y     = math.sqrt(4.0 * math.pi * alpha)

        # Tree-level: v_φ from M_χ = y v_φ / √2
        self.v_phi = math.sqrt(2.0) * self.m_chi / self.y
        self.lam4  = self.m_phi**2 / self.v_phi**2
        self.mu2   = self.m_phi**2
        self.mu_ren = self.m_chi
        self.T_fo  = self.m_chi / 20.0

    def M_chi_sq(self, phi):
        return 0.5 * self.y**2 * phi**2

    def M_phi_sq(self, phi):
        return -self.mu2 + 3.0 * self.lam4 * phi**2

    def V_tree(self, phi):
        return -0.5 * self.mu2 * phi**2 + 0.25 * self.lam4 * phi**4

    def V_CW(self, phi):
        Mq2 = self.M_chi_sq(phi)
        if Mq2 < 1e-60:
            return 0.0
        log_term = np.log(Mq2 / self.mu_ren**2) - 1.5
        return -float(N_F_MAJORANA) / (64.0 * np.pi**2) * Mq2**2 * log_term

    def V_thermal(self, phi, T):
        if T < 1e-30:
            return 0.0
        Mphi2 = self.M_phi_sq(phi)
        Mchi2 = self.M_chi_sq(phi)
        xb = math.sqrt(max(abs(Mphi2), 0.0)) / T
        xf = math.sqrt(max(Mchi2, 0.0)) / T
        jb = J_thermal(xb, boson=True)
        jf = J_thermal(xf, boson=False)
        pref = T**4 / (2.0 * np.pi**2)
        return pref * (jb - float(G_CHI) * jf)

    def V_eff(self, phi, T):
        return self.V_tree(phi) + self.V_CW(phi) + self.V_thermal(phi, T)

    def V_eff_array(self, phi_arr, T):
        return np.array([self.V_eff(p, T) for p in phi_arr])


# ══════════════════════════════════════════════════════════════════════
# PHASE TRANSITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def find_local_minima(phi_arr, V_arr):
    dV = np.diff(V_arr)
    minima = []
    for i in range(len(dV) - 1):
        if dV[i] < 0 and dV[i + 1] >= 0:
            minima.append((phi_arr[i + 1], V_arr[i + 1]))
    return minima


def scan_phase_transition(pot, n_T=300, n_phi=800):
    v = pot.v_phi
    phi_arr = np.linspace(0.0, 2.0 * v, n_phi)

    T_lo = pot.m_phi / 10.0
    T_hi = 10.0 * pot.m_chi
    T_arr = np.geomspace(T_lo, T_hi, n_T)[::-1]

    results = {
        "label": pot.label, "v_phi": v,
        "T_arr": [], "n_minima": [], "minima_phi": [], "minima_V": [],
        "T_c": None, "phi_broken_at_Tc": None, "first_order": False,
    }

    for T in T_arr:
        V_vals = pot.V_eff_array(phi_arr, T)
        lm = find_local_minima(phi_arr, V_vals)
        n_min = len(lm)

        results["T_arr"].append(T)
        results["n_minima"].append(n_min)
        results["minima_phi"].append([m[0] for m in lm])
        results["minima_V"].append([m[1] for m in lm])

        if n_min >= 2:
            lm_sorted = sorted(lm, key=lambda x: x[0])
            phi_sym, V_sym = lm_sorted[0]
            phi_brk, V_brk = lm_sorted[-1]
            delta_V = abs(V_sym - V_brk)
            V_depth = max(abs(V_sym), abs(V_brk), 1e-100)
            if delta_V / V_depth < 0.01:
                if results["T_c"] is None:
                    results["T_c"] = T
                    results["phi_broken_at_Tc"] = phi_brk
                    results["first_order"] = True

    return results


def compute_latent_heat(pot, T_c, phi_broken):
    dT = T_c * 1e-4
    V_sym_p = pot.V_eff(0.0, T_c + dT)
    V_brk_p = pot.V_eff(phi_broken, T_c + dT)
    V_sym_m = pot.V_eff(0.0, T_c - dT)
    V_brk_m = pot.V_eff(phi_broken, T_c - dT)
    d_deltaV_dT = ((V_sym_p - V_brk_p) - (V_sym_m - V_brk_m)) / (2.0 * dT)
    delta_V = pot.V_eff(0.0, T_c) - pot.V_eff(phi_broken, T_c)
    return T_c * d_deltaV_dT - delta_V


def compute_entropy_dilution(T_c, L, g_star_s=G_STAR_S):
    s_before = (2.0 * np.pi**2 / 45.0) * g_star_s * T_c**3
    D = 1.0 + L / (T_c * s_before)
    return D, s_before


def s_wave_sv(m_chi, alpha):
    sv_gev2 = math.pi * alpha**2 / (4.0 * m_chi**2)
    return sv_gev2 * GEV2_TO_CM2 * 3e10

def relic_naive(m_chi, alpha):
    return PLANCK_SV / s_wave_sv(m_chi, alpha) * OMEGA_TARGET


# ══════════════════════════════════════════════════════════════════════
# ANALYTIC ESTIMATES (before full numerical scan)
# ══════════════════════════════════════════════════════════════════════

def analytic_estimates(pot):
    """Quick analytic check: CW regime? T_c? Supercooling?"""
    y4_16pi2 = pot.y**4 / (16.0 * np.pi**2)
    cw_over_tree = y4_16pi2 / max(pot.lam4, 1e-100)

    # Thermal mass: Π = y²T²/12. Symmetry restored when Π T² > m_φ²
    T_c_est = pot.m_phi * math.sqrt(12.0) / pot.y

    # Vacuum energy difference
    delta_V = 0.25 * pot.mu2 * pot.v_phi**2

    # Radiation at T_c
    g_star = 5.0  # dark sector: 1 scalar + 2 fermion dof ~ 5 effective
    rho_rad = (np.pi**2 / 30.0) * g_star * T_c_est**4

    # Reheat temperature
    T_rh = (30.0 * delta_V / (np.pi**2 * g_star))**0.25

    return {
        "y4_16pi2": y4_16pi2,
        "lam4": pot.lam4,
        "CW_over_tree": cw_over_tree,
        "T_c_est_GeV": T_c_est,
        "T_fo_GeV": pot.T_fo,
        "delta_V_GeV4": delta_V,
        "rho_rad_GeV4": rho_rad,
        "supercooling_ratio": delta_V / max(rho_rad, 1e-100),
        "T_rh_GeV": T_rh,
        "T_rh_over_Tc": T_rh / max(T_c_est, 1e-100),
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("PI-8: FIRST-ORDER PHASE TRANSITION → ENTROPY DILUTION → RELIC FIX")
    print("     (standalone version — no project dependencies)")
    print("=" * 72)

    for label in TEST_BPS:
        bp = BENCHMARKS[label]
        m_chi = bp["m_chi_GeV"]
        m_phi_MeV = bp["m_phi_MeV"]
        alpha = bp["alpha"]
        y = math.sqrt(4.0 * math.pi * alpha)

        print(f"\n{'━' * 72}")
        print(f"  BENCHMARK: {label}")
        print(f"{'━' * 72}")
        print(f"  m_χ = {m_chi*1e3:.2f} MeV,  m_φ = {m_phi_MeV:.2f} MeV,  "
              f"α = {alpha:.4e},  y = {y:.4f}")

        pot = PhiPotential(label, m_chi, m_phi_MeV, alpha)
        print(f"  v_φ = {pot.v_phi:.4e} GeV,  λ₄ = {pot.lam4:.4e}")

        # ── analytic estimates ────────────────────────────────────────
        est = analytic_estimates(pot)
        print(f"\n  ANALYTIC ESTIMATES:")
        print(f"    y⁴/(16π²) = {est['y4_16pi2']:.4e}")
        print(f"    λ₄        = {est['lam4']:.4e}")
        print(f"    CW/tree   = {est['CW_over_tree']:.0f}×  "
              f"{'→ CW regime ✅' if est['CW_over_tree'] > 10 else '→ tree regime'}")
        print(f"    T_c (est) = {est['T_c_est_GeV']*1e3:.1f} MeV")
        print(f"    T_fo      = {est['T_fo_GeV']*1e3:.1f} MeV")
        print(f"    T_c < T_fo? {'✅ YES — timing OK' if est['T_c_est_GeV'] < est['T_fo_GeV'] else '❌ NO — PT before freeze-out'}")
        print(f"    ΔV/ρ_rad  = {est['supercooling_ratio']:.0f}  "
              f"{'⚠️ supercooled' if est['supercooling_ratio'] > 10 else '✅ mild'}")
        print(f"    T_rh      = {est['T_rh_GeV']*1e3:.1f} MeV")
        print(f"    T_rh/T_c  = {est['T_rh_over_Tc']:.2f}  "
              f"{'⚠️ reheat > T_c' if est['T_rh_over_Tc'] > 1 else '✅ reheat < T_c'}")

        # ── s-wave relic ──────────────────────────────────────────────
        sv = s_wave_sv(m_chi, alpha)
        omega_n = relic_naive(m_chi, alpha)
        print(f"\n  RELIC (s-wave only):")
        print(f"    ⟨σv⟩      = {sv:.3e} cm³/s")
        print(f"    ⟨σv⟩/Planck = {sv/PLANCK_SV:.3f}")
        print(f"    Ωh²(naive)  = {omega_n:.4f}  (target = {OMEGA_TARGET})")
        print(f"    D needed    = {omega_n / OMEGA_TARGET:.2f}")

        # ── numerical V_eff scan ──────────────────────────────────────
        print(f"\n  SCANNING V_eff(φ, T) numerically ...")
        res = scan_phase_transition(pot, n_T=300, n_phi=800)

        # Show key temperatures
        print(f"\n  {'T [MeV]':>10}  {'T/m_χ':>8}  {'# min':>6}  "
              f"{'φ_min locations [GeV]'}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*30}")
        step = max(1, len(res["T_arr"]) // 15)
        for i in range(0, len(res["T_arr"]), step):
            T = res["T_arr"][i]
            nm = res["n_minima"][i]
            phis = res["minima_phi"][i]
            phi_str = ", ".join(f"{p:.3e}" for p in phis) if phis else "none"
            print(f"  {T*1e3:>10.2f}  {T/m_chi:>8.4f}  {nm:>6d}  {phi_str}")

        # ── verdict ───────────────────────────────────────────────────
        print(f"\n  {'─'*60}")
        if res["first_order"]:
            T_c = res["T_c"]
            phi_brk = res["phi_broken_at_Tc"]
            print(f"  ✅ FIRST-ORDER PT DETECTED")
            print(f"     T_c = {T_c*1e3:.4f} MeV  ({T_c/m_chi:.4f} m_χ)")
            print(f"     φ_broken(T_c) = {phi_brk:.4e} GeV")

            L = compute_latent_heat(pot, T_c, phi_brk)
            D, s_bef = compute_entropy_dilution(T_c, L)
            omega_c = omega_n / D
            print(f"     Latent heat L = {L:.4e} GeV⁴")
            print(f"     Dilution D    = {D:.4f}")
            print(f"     Ωh²(corrected) = {omega_c:.4f}")
            print(f"     Target          = {OMEGA_TARGET}")
            ok = abs(omega_c - OMEGA_TARGET) / OMEGA_TARGET < 0.1
            print(f"     {'✅ MATCH' if ok else '✗ NO MATCH'}")
        else:
            print(f"  ✗ NO FIRST-ORDER PT FOUND")
            print(f"    Single minimum at all temperatures.")

    # ── summary table ─────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print("SUMMARY TABLE")
    print(f"{'═'*72}")
    print(f"  {'BP':>12}  {'CW/tree':>8}  {'T_c[MeV]':>9}  {'T_fo[MeV]':>9}  "
          f"{'1st-ord?':>8}  {'D':>8}  {'Ωh²':>8}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}")

    for label in TEST_BPS:
        bp = BENCHMARKS[label]
        pot = PhiPotential(label, bp["m_chi_GeV"], bp["m_phi_MeV"], bp["alpha"])
        est = analytic_estimates(pot)
        omega_n = relic_naive(bp["m_chi_GeV"], bp["alpha"])
        res = scan_phase_transition(pot, n_T=200, n_phi=600)

        if res["first_order"]:
            L = compute_latent_heat(pot, res["T_c"], res["phi_broken_at_Tc"])
            D, _ = compute_entropy_dilution(res["T_c"], L)
            omega_c = omega_n / D
            print(f"  {label:>12}  {est['CW_over_tree']:>8.0f}  "
                  f"{est['T_c_est_GeV']*1e3:>9.1f}  {est['T_fo_GeV']*1e3:>9.1f}  "
                  f"{'YES':>8}  {D:>8.3f}  {omega_c:>8.4f}")
        else:
            print(f"  {label:>12}  {est['CW_over_tree']:>8.0f}  "
                  f"{est['T_c_est_GeV']*1e3:>9.1f}  {est['T_fo_GeV']*1e3:>9.1f}  "
                  f"{'NO':>8}  {'—':>8}  {omega_n:>8.4f}")

    print(f"\n  Target: Ωh² = {OMEGA_TARGET}")
    print(f"\n{'═'*72}")


if __name__ == "__main__":
    main()
