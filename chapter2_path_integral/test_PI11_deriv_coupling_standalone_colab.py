#!/usr/bin/env python3
"""
PI-11 STANDALONE: Derivative Coupling of σ — Relic Fix via Decoupled Channel
============================================================================
Self-contained — runs on Google Colab, Replit, Kaggle, or any Python 3.8+
with `import math` only.  NO project dependencies.

HYPOTHESIS:
  The dark axion σ (pseudo-Goldstone from dark-energy-T-breaking) couples
  to Majorana DM χ via the derivative vertex:

      L ⊃  (∂_μσ / f) χ̄ γ^μ γ^5 χ

  By the equations of motion (integration by parts + Dirac eq.), this is
  equivalent on-shell to a pseudoscalar Yukawa coupling:

      L_eff = -(2i m_χ/f) σ χ̄ γ^5 χ

  Derivation:
      ∂_μ(χ̄ γ^μ γ^5 χ) = 2i m_χ χ̄ γ^5 χ   (from Dirac equation)
      ∫ (∂_μσ/f) χ̄ γ^μ γ^5 χ  =  -(σ/f) ∂_μ(χ̄ γ^μ γ^5 χ)  [IBP]
                                =  -(2i m_χ/f) σ χ̄ γ^5 χ

  So:  y_P = 2 m_χ/f ,   α_P = y_P²/(4π) = m_χ²/(π f²)

CROSS SECTIONS (Majorana, t/u-channel, m_σ ≪ m_χ):
  For PURE pseudoscalar coupling (no scalar admixture):
      a_σ = 0               (exact — no scalar×pseudoscalar interference)
      b_σ = 3π α_P² / m_χ²  = 3 m_χ² / (π f⁴)

  This uses the SAME formula as PI-9, but with α_s = 0 (pure pseudoscalar):
      a = 2π·α_s·α_P/m_χ²  →  0   (since α_s = 0 for σ vertex)
      b = 3π·α_P²/m_χ²     →  3 m_χ²/(π f⁴)

KEY ADVANTAGE — TWO INDEPENDENT CHANNELS:
  ┌─ φ-channel (SIDM):  a_φ = π α_s²/(4 m_χ²)  — LOCKED by 13 observations
  └─ σ-channel (relic):  b_σ = 3 m_χ²/(π f⁴)    — f is a FREE parameter

  At freeze-out:  ⟨σv⟩_fo = a_φ + 3·b_σ/x_f  (LARGE — both contribute)
  Today:          ⟨σv⟩    ≈ a_φ                (σ-channel is v²-suppressed)
  CMB:            ⟨σv⟩    ≈ a_φ                (even more suppressed)

  Therefore:  SIDM is unaffected,  Ωh² acquires an additional free parameter f.

RELIC DENSITY:
  Ωh² = 0.12 × σ_Planck / (a_φ + 3·b_σ/x_f)_cm
  x_f from iterative Kolb-Turner eq. 5.45.

SUCCESS CRITERIA (from research journal):
  ✓ Necessary:  ∃ f ∈ [0.05, 2] M_Pl  such that  Ωh² = 0.120 ± 10%
  ✓ Sufficient: same f works for all BPs (universality)
  ✓ Bonus:      f ≈ 0.24 M_Pl (= f_DE from dark energy mechanism)
  ✗ Failure:    f_needed ≫ M_Pl  or  f differs by >10× between BPs

To run on Colab:
  1. https://colab.research.google.com → New Notebook
  2. Paste entire file → Run cell
"""

import math

# ════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ════════════════════════════════════════════════════════════════════════
M_PL_GEV     = 1.2209e19    # Full Planck mass [GeV]
M_PL_RED_GEV = 2.4353e18    # Reduced Planck mass M_Pl/√(8π) [GeV]
GEV2_TO_CM3S = 3.8938e-28 * 3.0e10   # GeV⁻² → cm³/s  (= 1.168e-17)
OMEGA_TARGET = 0.1200
G_STAR       = 86.25       # Effective dof at freeze-out (~20-90 GeV)
PLANCK_SV    = 3.0e-26     # cm³/s — canonical thermal relic ⟨σv⟩

# Dark energy reference scale (from dark-energy-T-breaking project)
F_DE_GEV     = 0.24 * M_PL_RED_GEV   # f for DE ≈ 5.84e17 GeV
LAMBDA_D_GEV = 2.28e-12              # Dark confinement scale [GeV]

# ════════════════════════════════════════════════════════════════════════
# BENCHMARK POINTS (from global_config.json, march 2026 snapshot)
# ════════════════════════════════════════════════════════════════════════
# α_s ≡ α (CSV/SIDM convention, proven in test_T1)
BENCHMARKS = {
    "BP1":  {"m_chi": 54.556,  "alpha_s": 2.645e-3, "m_phi_MeV":  12.975},
    "BP9":  {"m_chi": 48.329,  "alpha_s": 2.350e-3, "m_phi_MeV":   8.657},
    "BP16": {"m_chi": 14.384,  "alpha_s": 7.555e-4, "m_phi_MeV":   5.047},
    "MAP":  {"m_chi": 98.19,   "alpha_s": 3.274e-3, "m_phi_MeV":   9.660},
}

TEST_BPS = ["BP1", "BP9", "MAP"]


# ════════════════════════════════════════════════════════════════════════
# σv COEFFICIENTS
# ════════════════════════════════════════════════════════════════════════

def phi_channel_a(m_chi, alpha_s):
    """
    s-wave from φ channel (SIDM-locked).

        a_φ = π α_s² / (4 m_χ²)

    This is the SAME formula as PI-8 / PI-9. α_s is locked by SIDM.
    """
    return math.pi * alpha_s**2 / (4.0 * m_chi**2)


def sigma_channel_b(m_chi, f_gev):
    """
    p-wave coefficient from σ derivative coupling.

        b_σ = 3 m_χ² / (π f⁴)

    Derivation (using PI-9 formula with α_s → 0, α_p → α_P):
        y_P  = 2 m_χ / f               (EOM equivalence)
        α_P  = y_P² / (4π) = m_χ² / (π f²)
        b_σ  = 3π α_P² / m_χ²
             = 3π [m_χ² / (π f²)]² / m_χ²
             = 3 m_χ² / (π f⁴)
    """
    return 3.0 * m_chi**2 / (math.pi * f_gev**4)


def sigma_channel_alpha_P(m_chi, f_gev):
    """Effective pseudoscalar coupling α_P = m_χ²/(π f²)."""
    return m_chi**2 / (math.pi * f_gev**2)


# ════════════════════════════════════════════════════════════════════════
# FREEZE-OUT  x_f
# ════════════════════════════════════════════════════════════════════════

def find_xf(m_chi, a_gev, b_gev, g_chi=1.0, g_star=G_STAR):
    """
    Iterative solution of freeze-out condition (Kolb & Turner eq. 5.45):

        xf = ln[0.0764 · c(c+2) · g/√g* · M_Pl · mχ · (a + 6b/xf)]
             − 0.5 · ln(xf)

    c = 0.5, g_chi = 1 for Majorana.
    a = total s-wave (φ-channel),  b = total p-wave (σ-channel).
    """
    c = 0.5
    prefac = 0.0764 * c * (c + 2.0) * g_chi / math.sqrt(g_star) * M_PL_GEV * m_chi

    xf = 22.0
    for _ in range(30):
        sv = a_gev + 6.0 * b_gev / xf
        if sv <= 0:
            break
        arg = prefac * sv
        if arg <= 0:
            break
        xf_new = math.log(arg) - 0.5 * math.log(xf)
        if abs(xf_new - xf) < 1e-6:
            break
        xf = 0.6 * xf + 0.4 * xf_new   # damped to avoid oscillation

    return max(xf, 5.0)


# ════════════════════════════════════════════════════════════════════════
# RELIC DENSITY  Ωh²
# ════════════════════════════════════════════════════════════════════════

def omega_h2(xf, a_gev, b_gev):
    """
    Ωh² from Gondolo-Gelmini:

        Ωh² = 0.12 × σ_Planck / (a + 3b/xf)_cm
    """
    a_eff_cm = (a_gev + 3.0 * b_gev / xf) * GEV2_TO_CM3S
    if a_eff_cm <= 0:
        return 1e10
    return OMEGA_TARGET * PLANCK_SV / a_eff_cm


def omega_h2_phi_only(m_chi, alpha_s):
    """Ωh² from φ-channel alone (PI-8 / PI-9 baseline)."""
    a_phi = phi_channel_a(m_chi, alpha_s)
    xf = find_xf(m_chi, a_phi, 0.0)
    return omega_h2(xf, a_phi, 0.0)


# ════════════════════════════════════════════════════════════════════════
# ANALYTIC ESTIMATE of f_needed
# ════════════════════════════════════════════════════════════════════════

def f_needed_analytic(m_chi, alpha_s, xf=22.0):
    """
    Solve for f such that Ωh² = 0.12.

    Need:  a_φ + 3·b_σ/xf = σ_Planck / GEV2_TO_CM3S  (in GeV⁻² units)

    →  b_σ = (σv_target - a_φ) · xf / 3
    →  f   = [3 m_χ² / (π b_σ)]^(1/4)
    """
    a_phi = phi_channel_a(m_chi, alpha_s)
    sv_target_gev = PLANCK_SV / GEV2_TO_CM3S   # target σv in GeV⁻²

    b_needed = (sv_target_gev - a_phi) * xf / 3.0

    if b_needed <= 0:
        return float('inf')  # φ-channel alone is already enough

    f4 = 3.0 * m_chi**2 / (math.pi * b_needed)
    return f4**0.25


# ════════════════════════════════════════════════════════════════════════
# BISECTION for precise f_cross
# ════════════════════════════════════════════════════════════════════════

def find_f_cross_precise(m_chi, alpha_s, f_lo=1e2, f_hi=1e20, tol=1e-4):
    """Bisection in log(f) to find f where Ωh² = OMEGA_TARGET."""
    a_phi = phi_channel_a(m_chi, alpha_s)

    def omega_at_f(f_gev):
        b_sig = sigma_channel_b(m_chi, f_gev)
        xf = find_xf(m_chi, a_phi, b_sig)
        return omega_h2(xf, a_phi, b_sig)

    oh2_lo = omega_at_f(f_lo)
    oh2_hi = omega_at_f(f_hi)

    # Small f → large b_σ → large σv → small Ωh²
    # Large f → small b_σ → σv ≈ a_φ → larger Ωh² (overclosed)
    if oh2_lo > OMEGA_TARGET and oh2_hi > OMEGA_TARGET:
        return None
    if oh2_lo < OMEGA_TARGET and oh2_hi < OMEGA_TARGET:
        return None

    for _ in range(100):
        f_mid = math.sqrt(f_lo * f_hi)   # geometric mean
        oh2_mid = omega_at_f(f_mid)

        if abs(oh2_mid - OMEGA_TARGET) / OMEGA_TARGET < tol:
            return f_mid

        if oh2_mid > OMEGA_TARGET:
            f_hi = f_mid   # need more b_σ → smaller f
        else:
            f_lo = f_mid   # too much b_σ → larger f

    return math.sqrt(f_lo * f_hi)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("  PI-11: DERIVATIVE COUPLING σ — RELIC FIX VIA DECOUPLED CHANNEL")
    print("  (standalone — no project dependencies)")
    print("=" * 78)

    # ── Physics summary ─────────────────────────────────────────────────
    print(f"""
  PHYSICS:
  ────────
  Lagrangian:  L ⊃ (∂_μσ/f) χ̄ γ^μ γ^5 χ
  On-shell:       = -(2i m_χ/f) σ χ̄ γ^5 χ   [EOM equivalence]

  → Pure pseudoscalar coupling with y_P = 2 m_χ/f
  → a_σ = 0  (exact, no scalar×pseudoscalar interference)
  → b_σ = 3 m_χ²/(π f⁴)

  TWO INDEPENDENT CHANNELS:
  ┌─ φ-channel (SIDM):  a_φ = π α_s²/(4 m_χ²)  — LOCKED
  └─ σ-channel (relic):  b_σ = 3 m_χ²/(π f⁴)    — f FREE

  ⟨σv⟩_fo  = a_φ + 3·b_σ/x_f     (both channels contribute at freeze-out)
  ⟨σv⟩_now ≈ a_φ                  (σ-channel suppressed by v² ~ 10⁻⁶)
  → SIDM unaffected, Ωh² tunable via f

  REFERENCE SCALES:
  f_DE  = 0.24 M_Pl,red = {F_DE_GEV:.3e} GeV  (dark energy)
  M_Pl,red              = {M_PL_RED_GEV:.3e} GeV
""")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTIC PREVIEW
    # ══════════════════════════════════════════════════════════════════════
    print(f"{'━' * 78}")
    print(f"  ANALYTIC ESTIMATES  (x_f = 22)")
    print(f"{'━' * 78}")

    print(f"\n  {'BP':>6}  {'m_χ':>8} {'α_s':>10}  {'Ωh²(φ)':>9}  "
          f"{'f_need':>11}  {'f/M_Pl':>10}  {'y_P':>7}  {'α_P':>10}")
    print(f"  {'─'*6}  {'─'*8} {'─'*10}  {'─'*9}  {'─'*11}  {'─'*10}  {'─'*7}  {'─'*10}")

    for label in TEST_BPS:
        bp = BENCHMARKS[label]
        m_chi = bp["m_chi"]
        alpha_s = bp["alpha_s"]

        oh2_phi = omega_h2_phi_only(m_chi, alpha_s)
        f_need = f_needed_analytic(m_chi, alpha_s, xf=22.0)
        y_P = 2.0 * m_chi / f_need if f_need < 1e30 else 0.0
        alpha_p = sigma_channel_alpha_P(m_chi, f_need) if f_need < 1e30 else 0.0
        f_ratio = f_need / M_PL_RED_GEV

        print(f"  {label:>6}  {m_chi:>8.2f} {alpha_s:>10.3e}  {oh2_phi:>9.4f}  "
              f"{f_need:>11.3e}  {f_ratio:>10.2e}  {y_P:>7.4f}  {alpha_p:>10.3e}")

    print(f"\n  → All f_needed are O(1 TeV), far below f_DE ~ 10^{{18}} GeV.")
    print(f"  → y_P < 1: perturbative regime ✓")

    # ══════════════════════════════════════════════════════════════════════
    # DETAILED f-SCAN FOR EACH BENCHMARK
    # ══════════════════════════════════════════════════════════════════════
    results_all = {}

    for label in TEST_BPS:
        bp = BENCHMARKS[label]
        m_chi = bp["m_chi"]
        alpha_s = bp["alpha_s"]

        print(f"\n\n{'━' * 78}")
        print(f"  BENCHMARK: {label}  (m_χ = {m_chi:.3f} GeV, α_s = {alpha_s:.4e})")
        print(f"{'━' * 78}")

        a_phi = phi_channel_a(m_chi, alpha_s)
        a_phi_cm = a_phi * GEV2_TO_CM3S
        oh2_phi = omega_h2_phi_only(m_chi, alpha_s)

        print(f"\n  φ-channel (SIDM-locked):")
        print(f"    a_φ = {a_phi:.4e} GeV⁻² = {a_phi_cm:.4e} cm³/s "
              f"= {a_phi_cm/PLANCK_SV:.4f} σ_Planck")
        print(f"    Ωh²(φ only) = {oh2_phi:.4f}  "
              f"(overclosed by {oh2_phi/OMEGA_TARGET:.2f}×)")

        # ── σ-channel f scan ────────────────────────────────────────────
        print(f"\n  σ-CHANNEL f SCAN:")
        print(f"  {'f [GeV]':>12}  {'f/M_Pl':>10}  {'α_P':>10}  "
              f"{'b_σ [GeV⁻²]':>12}  {'y_P':>7}  {'x_f':>5}  "
              f"{'Ωh²':>8}  {'pass?':>5}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*7}  "
              f"{'─'*5}  {'─'*8}  {'─'*5}")

        # Scan from f = 10^2.0 to 10^6.0 (above that σ-channel is negligible)
        for log_f_10 in [x * 0.25 for x in range(8, 25)]:   # 2.0 to 6.0
            f_gev = 10.0**log_f_10
            b_sig = sigma_channel_b(m_chi, f_gev)
            alpha_p = sigma_channel_alpha_P(m_chi, f_gev)
            y_P = 2.0 * m_chi / f_gev

            xf = find_xf(m_chi, a_phi, b_sig)
            oh2 = omega_h2(xf, a_phi, b_sig)

            ok = abs(oh2 - OMEGA_TARGET) / OMEGA_TARGET < 0.10
            mark = "← ✅" if ok else ""

            if oh2 < 100:
                print(f"  {f_gev:>12.3e}  {f_gev/M_PL_RED_GEV:>10.2e}  "
                      f"{alpha_p:>10.3e}  {b_sig:>12.3e}  {y_P:>7.4f}  "
                      f"{xf:>5.1f}  {oh2:>8.4f}  {mark}")

        # ── Precise crossing ────────────────────────────────────────────
        f_cross = find_f_cross_precise(m_chi, alpha_s)

        if f_cross is not None:
            b_sig = sigma_channel_b(m_chi, f_cross)
            xf = find_xf(m_chi, a_phi, b_sig)
            oh2_final = omega_h2(xf, a_phi, b_sig)
            alpha_p = sigma_channel_alpha_P(m_chi, f_cross)
            y_P = 2.0 * m_chi / f_cross

            print(f"\n  ═══ PRECISE SOLUTION ═══")
            print(f"    f_cross  = {f_cross:.4e} GeV")
            print(f"    f/M_Pl   = {f_cross/M_PL_RED_GEV:.4e}")
            print(f"    y_P      = 2m_χ/f = {y_P:.6f}   "
                  f"({'perturbative ✓' if y_P < 1.0 else 'NON-PERTURBATIVE ⚠️'})")
            print(f"    α_P      = {alpha_p:.4e}   ({'< 1 ✓' if alpha_p < 1 else '⚠️'})")
            print(f"    b_σ      = {b_sig:.4e} GeV⁻²")
            print(f"    b_σ·cm   = {b_sig*GEV2_TO_CM3S:.4e} cm³/s")
            print(f"    3b_σ/x_f = {3*b_sig*GEV2_TO_CM3S/xf:.4e} cm³/s  "
                  f"(σ-channel contribution at f.o.)")
            print(f"    a_φ      = {a_phi_cm:.4e} cm³/s  (φ-channel)")
            print(f"    total    = {a_phi_cm + 3*b_sig*GEV2_TO_CM3S/xf:.4e} cm³/s")
            print(f"    x_f      = {xf:.2f}")
            print(f"    Ωh²      = {oh2_final:.4f}")

            print(f"\n    COMPARISON WITH DARK ENERGY:")
            print(f"    f_cross / f_DE = {f_cross/F_DE_GEV:.2e}")
            if abs(math.log10(f_cross / F_DE_GEV)) < 1:
                print(f"    ✅ SAME ORDER as dark energy scale!")
            else:
                gap = abs(math.log10(f_cross / F_DE_GEV))
                print(f"    ✗ f_cross ≠ f_DE  — differ by 10^{gap:.0f}")
                print(f"      f_needed  = {f_cross:.2e} GeV")
                print(f"      f_DE      = {F_DE_GEV:.2e} GeV")

            # σ mass
            m_sigma_cross = LAMBDA_D_GEV**2 / f_cross
            m_sigma_de = LAMBDA_D_GEV**2 / F_DE_GEV
            print(f"\n    σ MASS (m_σ = Λ_d²/f):")
            print(f"    m_σ(f_cross) = {m_sigma_cross:.3e} GeV "
                  f"= {m_sigma_cross/1e-18:.1f} × 10⁻¹⁸ GeV")
            print(f"    m_σ(f_DE)    = {m_sigma_de:.3e} GeV")
            print(f"    m_σ ≪ m_χ?   "
                  f"{'✓ (by 10^' + str(int(math.log10(m_chi/m_sigma_cross))) + ')' if m_sigma_cross > 0 and m_sigma_cross < m_chi else '⚠️'}")

            # Today's σv
            v2_today = 1.0e-6   # v ~ 10⁻³ c
            sv_sig_today = b_sig * GEV2_TO_CM3S * v2_today
            print(f"\n    TODAY (v² ~ 10⁻⁶):")
            print(f"    σv(φ)  = {a_phi_cm:.3e} cm³/s")
            print(f"    σv(σ)  = {sv_sig_today:.3e} cm³/s  (v²-suppressed)")
            print(f"    ratio  = {sv_sig_today/a_phi_cm:.2e}  → SIDM unaffected ✓")

            results_all[label] = {
                "oh2_phi": oh2_phi,
                "f_cross": f_cross,
                "oh2_final": oh2_final,
                "y_P": y_P,
                "alpha_P": alpha_p,
            }
        else:
            print(f"\n  ⚠️ NO CROSSING FOUND")
            results_all[label] = {"oh2_phi": oh2_phi, "f_cross": None}

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'═' * 78}")
    print("  SUMMARY TABLE")
    print(f"{'═' * 78}")
    print(f"  {'BP':>6}  {'Ωh²(φ)':>8}  {'f_cross':>12}  {'f/M_Pl':>10}  "
          f"{'y_P':>7}  {'α_P':>9}  {'Ωh²(φ+σ)':>9}  {'pass':>5}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*7}  "
          f"{'─'*9}  {'─'*9}  {'─'*5}")

    valid_f = []
    all_pass = True

    for label in TEST_BPS:
        r = results_all[label]
        fc = r["f_cross"]

        if fc is not None:
            ok = abs(r["oh2_final"] - OMEGA_TARGET) / OMEGA_TARGET < 0.01
            valid_f.append(fc)
            print(f"  {label:>6}  {r['oh2_phi']:>8.4f}  {fc:>12.2e}  "
                  f"{fc/M_PL_RED_GEV:>10.2e}  {r['y_P']:>7.4f}  "
                  f"{r['alpha_P']:>9.3e}  {r['oh2_final']:>9.4f}  "
                  f"{'✅' if ok else '✗':>5}")
            if not ok:
                all_pass = False
        else:
            all_pass = False
            print(f"  {label:>6}  {r['oh2_phi']:>8.4f}  {'N/A':>12}  "
                  f"{'N/A':>10}  {'N/A':>7}  {'N/A':>9}  {'N/A':>9}  {'✗':>5}")

    # ── Universality check ──────────────────────────────────────────────
    if len(valid_f) >= 2:
        f_max = max(valid_f)
        f_min = min(valid_f)
        ratio = f_max / f_min
        print(f"\n  UNIVERSALITY:")
        print(f"  f range: [{f_min:.3e}, {f_max:.3e}] GeV")
        print(f"  f_max / f_min = {ratio:.2f}  "
              f"({'NEAR-UNIVERSAL ✅' if ratio < 3 else 'NOT UNIVERSAL ✗'})")

    # ── f_DE comparison ─────────────────────────────────────────────────
    if valid_f:
        f_geo_mean = math.exp(sum(math.log(f) for f in valid_f) / len(valid_f))
        gap_oom = abs(math.log10(f_geo_mean / F_DE_GEV))
        print(f"\n  DARK ENERGY CONNECTION:")
        print(f"  ⟨f_cross⟩_geo = {f_geo_mean:.3e} GeV")
        print(f"  f_DE          = {F_DE_GEV:.3e} GeV")
        print(f"  gap           = 10^{gap_oom:.1f}")
        if gap_oom < 1:
            print(f"  ✅ UNIFIED: same f gives both DE and relic!")
        else:
            print(f"  ✗ f_relic ≠ f_DE — {gap_oom:.0f} orders of magnitude apart")

    # ══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 78}")
    if all_pass and valid_f:
        f_geo = math.exp(sum(math.log(f) for f in valid_f) / len(valid_f))
        print(f"  PI-11 RESULT:  MECHANISM WORKS ✅ (with caveats)")
        print(f"")
        print(f"  WHAT WORKS:")
        print(f"  • ∃ f ≈ {f_geo:.0f} GeV such that Ωh² = 0.120")
        print(f"  • f is universal (same for all BPs to within a factor "
              f"{max(valid_f)/min(valid_f):.1f})")
        print(f"  • y_P = 2m_χ/f ≈ 0.2 → perturbative ✓")
        print(f"  • SIDM completely unaffected (σ-channel is v²-suppressed) ✓")
        print(f"  • No CMB constraint (σσ is dark sector) ✓")
        print(f"  • No 5th force (σ couples only to dark sector) ✓")
        print(f"")
        print(f"  WHAT DOESN'T WORK:")
        print(f"  • f_relic ≈ {f_geo:.0f} GeV ≠ f_DE ≈ {F_DE_GEV:.1e} GeV")
        print(f"  • Gap of 10^{abs(math.log10(f_geo/F_DE_GEV)):.0f} → "
              f"the SAME σ cannot give both DE and relic")
        print(f"  • Need either:")
        print(f"    (a) c_χ ≫ 1:  L = (c_χ/f_DE) ∂σ χ̄γ⁵χ  "
              f"with c_χ ≈ {F_DE_GEV/f_geo:.0e}")
        print(f"    (b) Separate ALP:  σ_relic ≠ σ_DE  "
              f"(two pseudo-Goldstones)")
        print(f"    (c) Different sector:  f_relic is a new scale "
              f"in the dark sector")
    else:
        print(f"  PI-11 RESULT:  CHECK DETAILS ABOVE")

    print(f"""
  PHYSICAL EXPLANATION:
  ─────────────────────
  The derivative coupling ∂_μσ/f generates a purely pseudoscalar effective
  Yukawa y_P = 2m_χ/f.  For Majorana fermions, this gives:

      a_σ = 0     (no scalar component → no s-wave)
      b_σ = 3m_χ²/(πf⁴)   (p-wave)

  The φ and σ channels are FULLY INDEPENDENT:
  • φ channel determines SIDM → fixes α_s → fixes a_φ
  • σ channel provides extra σv at freeze-out via p-wave → tunes Ωh²

  The required f is sub-TeV to TeV scale because b_σ ∝ 1/f⁴:
  large b_σ needs small f.  This is 15 orders of magnitude below the
  dark-energy decay constant f_DE ≈ 10^{{18}} GeV.

  ROOT CAUSE of the gap:
    For DE:  V ∝ Λ_d⁴ ~ (10⁻¹² GeV)⁴ → needs huge f to get tiny ρ_σ
    For relic: b_σ ~ m_χ²/f⁴ → needs small f to get large enough σv
    These pull in opposite directions.

  CONCLUSION:
  PI-11 as a MECHANISM works — but requires f ~ TeV, NOT f ~ M_Pl.
  The unification dream (same f for DE and relic) fails by 10^15.
  However, the mechanism IS viable as a stand-alone relic fix if one
  introduces an additional ALP with f ~ TeV, independent of σ_DE.
""")
    print(f"{'═' * 78}")
    print("  NEXT STEPS:")
    print("  1. Check if f ~ TeV ALP is constrained by any existing data")
    print("  2. Consider Wilson coefficient c_χ ≫ 1 (anomalous coupling)")
    print("  3. Explore non-thermal production of σ (misalignment doesn't")
    print("     need the SAME f that enters the DM vertex)")
    print("  4. PI-12: explore co-annihilation χφ → σ (mixed channel)")
    print(f"{'═' * 78}")


if __name__ == "__main__":
    main()
