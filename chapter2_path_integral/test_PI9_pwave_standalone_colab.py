#!/usr/bin/env python3
"""
PI-9 STANDALONE: P-wave Suppression via T-breaking (θ=19.47°) — Relic Fix?
============================================================================
Self-contained — runs on Google Colab, Replit, Kaggle, or any Python 3.8+
with numpy only.  NO project dependencies.

HYPOTHESIS:
  The universal angle θ_relic = arctan(1/√8) = 19.47°  from A₄ group theory
  introduces a pseudoscalar coupling y_p alongside the scalar y_s:

      y_s = y·cos(θ),   y_p = y·sin(θ)
      αs  = y_s²/4π = α       (SIDM constraint — fixes αs)
      αp  = y_p²/4π = α·tan²θ = α/8

  For χχ→φφ (Majorana, t/u-channel, mφ≪mχ):

      σv = a + b·v²

  where a = s-wave,  b·v² = p-wave (from pseudoscalar vertex).

  At freeze-out : ⟨v²⟩ ≈ 6/xf ≈ 0.27  →  σv_fo = a + 0.27b  (LARGE)
  Today (SIDM)  : ⟨v²⟩ ≈ 10⁻⁴          →  σv ≈ a             (SMALL)
  At CMB        : ⟨v²⟩ ≈ 10⁻⁶          →  σv ≈ a             (TINY)

  If b >> a:  Ωh² ≈ 0.12 × σ_Planck / (a + 3b/xf)  is much smaller than
  the naive s-wave estimate.

CROSS SECTIONS (Majorana fermion, mφ≪mχ, light-mediator limit):
  s-wave from mixed scalar×pseudoscalar interference:
      a = 2π·αs·αp / mχ²                 [theory.md §3.2]
          = π·αs² / (4mχ²)               [at θ_relic: same as pure scalar]
  p-wave from pure pseudoscalar (Majorana factor ×4 vs Dirac):
      b = 3π·αp² / mχ²                   [Jungman et al., Phys.Rept.267]

RELIC DENSITY (Kolb & Turner calibrated to PI-8 naive formula):
  Ωh²  ≈ 0.12 × σ_Planck / (a_cm  +  3·b_cm / xf)
  xf from iterative freeze-out condition (Kolb & Turner eq. 5.45).

To run on Colab:
  1. https://colab.research.google.com → New Notebook
  2. Paste entire file → Run cell
"""

import math

# ════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ════════════════════════════════════════════════════════════════════════
M_PL_GEV    = 1.2209e19    # Full Planck mass [GeV]
GEV2_TO_CM3S = 3.8938e-28 * 3.0e10   # GeV⁻² → cm³/s
OMEGA_TARGET = 0.1200
G_STAR       = 86.25      # Effective dof at freeze-out (~20-90 GeV)
PLANCK_SV    = 3.0e-26    # cm³/s — canonical thermal relic normalization

# Universal T-breaking angle from A₄ Clebsch-Gordan (proven in dark-energy-T-breaking)
THETA_RELIC  = math.atan(1.0 / math.sqrt(8.0))   # 19.47°
TAN2_THETA   = 1.0 / 8.0                          # tan²(θ_relic) = αp/αs

# ════════════════════════════════════════════════════════════════════════
# BENCHMARK POINTS (from global_config.json, march 2026 snapshot)
# ════════════════════════════════════════════════════════════════════════
# αs = α (SIDM alpha — same as α in PI-8)
BENCHMARKS = {
    "BP1":  {"m_chi": 54.556,  "alpha_s": 2.645e-3, "m_phi_MeV":  12.975},
    "BP9":  {"m_chi": 48.329,  "alpha_s": 2.350e-3, "m_phi_MeV":   8.657},
    "BP16": {"m_chi": 14.384,  "alpha_s": 7.555e-4, "m_phi_MeV":   5.047},
    "MAP":  {"m_chi": 98.19,   "alpha_s": 3.274e-3, "m_phi_MeV":   9.660},
}

TEST_BPS = ["BP1", "BP9", "MAP"]


# ════════════════════════════════════════════════════════════════════════
# σv COEFFICIENTS  a, b
# ════════════════════════════════════════════════════════════════════════

def sv_coefficients(m_chi, alpha_s):
    """
    Returns (alpha_p, a_GeV, b_GeV, a_scalar_GeV) for χχ→φφ at θ_relic.

    a_scalar = pure s-wave at θ=0 (reference).
    At θ_relic: a = a_scalar  exactly (that's why θ was chosen).
    b << a  because αp = αs/8  →  b ∝ αp² ∝ αs²/64.
    """
    alpha_p   = alpha_s * TAN2_THETA          # αs / 8
    m2        = m_chi ** 2

    # s-wave coefficient (mixed coupling, Majorana)
    a_gev     = 2.0 * math.pi * alpha_s * alpha_p / m2    # = π αs² / (4 mχ²)

    # p-wave coefficient (pure pseudoscalar, Majorana ×4 vs Dirac)
    b_gev     = 3.0 * math.pi * alpha_p ** 2 / m2

    # Reference: pure scalar (θ=0)
    a_scalar  = math.pi * alpha_s ** 2 / (4.0 * m2)

    return alpha_p, a_gev, b_gev, a_scalar


# ════════════════════════════════════════════════════════════════════════
# FREEZE-OUT  xf
# ════════════════════════════════════════════════════════════════════════

def find_xf(m_chi, a_gev, b_gev, g_chi=1.0, g_star=G_STAR):
    """
    Iterative solution of freeze-out condition (Kolb & Turner eq. 5.45):

        xf = ln[0.0764 · c(c+2) · g/√g* · MPl · mχ · (a + 6b/xf)]
             − 0.5 · ln(xf)

    c = 0.5, g_chi = 1 for Majorana.
    Converges in ~5-8 iterations.
    """
    c       = 0.5
    prefac  = 0.0764 * c * (c + 2.0) * g_chi / math.sqrt(g_star) * M_PL_GEV * m_chi

    xf = 22.0
    for _ in range(20):
        sv     = a_gev + 6.0 * b_gev / xf
        xf_new = math.log(prefac * sv) - 0.5 * math.log(xf)
        if abs(xf_new - xf) < 1e-5:
            break
        xf = 0.6 * xf + 0.4 * xf_new   # damped to avoid oscillation

    return max(xf, 5.0)


# ════════════════════════════════════════════════════════════════════════
# RELIC DENSITY  Ωh²
# ════════════════════════════════════════════════════════════════════════

def omega_h2(xf, a_gev, b_gev):
    """
    Ωh² using the relic formula calibrated to the PI-8 naive result:

        Ωh² ≈ 0.12 × σ_Planck / a_eff_cm
        a_eff  =  a  +  3b/xf          (the effective s-wave after p-wave fold-in)

    This is the Gondolo-Gelmini result for the standard approximation.
    For b=0 and σv = PLANCK_SV this returns exactly 0.12.
    """
    a_eff_cm = (a_gev + 3.0 * b_gev / xf) * GEV2_TO_CM3S
    return OMEGA_TARGET * PLANCK_SV / a_eff_cm


def omega_h2_naive(m_chi, alpha_s):
    """s-wave only Ωh² (same formula as PI-8, no p-wave)."""
    sv_gev = math.pi * alpha_s ** 2 / (4.0 * m_chi ** 2)
    sv_cm  = sv_gev * GEV2_TO_CM3S
    return OMEGA_TARGET * PLANCK_SV / sv_cm


# ════════════════════════════════════════════════════════════════════════
# ANALYTIC SUMMARY  (no loop needed)
# ════════════════════════════════════════════════════════════════════════

def analytic_pwave_summary():
    """
    At θ_relic the ratio b/a is universal (independent of mχ, α):

        a  = π αs² / (4 mχ²)
        b  = 3π αp² / mχ²  = 3π (αs/8)² / mχ²  = 3π αs² / (64 mχ²)

        b/a = [3π αs²/64 mχ²] / [π αs²/4 mχ²]  =  3·4/64  = 3/16

    p-wave boost at freeze-out (xf = 22):

        δσv/σv  =  6b/(a·xf)  =  6·(3/16)/22  =  18/(16·22)  ≈  5.1%

    Ωh² improvement: −5.1%  →  way too small for D=2.94 (needed: +194%).
    """
    b_over_a = 3.0 / 16.0
    xf_typical = 22.0
    boost_fo = 6.0 * b_over_a / xf_typical        # σv boost at f.o.
    omega_improvement = boost_fo / (1.0 + boost_fo)  # fractional Ωh² drop
    print(f"\n  ━━━  ANALYTIC PREVIEW (θ_relic, universal)  ━━━")
    print(f"  b/a           = 3/16 = {b_over_a:.4f}   (universal, independent of mχ, α)")
    print(f"  p-wave boost at xf={xf_typical:.0f}: {boost_fo*100:.1f}%")
    print(f"  Ωh² reduction:        {omega_improvement*100:.1f}%")
    print(f"  D needed (MAP):        2.94×  (≡ 194% reduction needed)")
    print(f"  → Conclusion BEFORE numerical scan: PI-9 fails by ×{2.94/(1-omega_improvement):.0f}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 74)
    print("  PI-9: P-WAVE FROM T-BREAKING θ=19.47°  — RELIC FIX?")
    print("  (standalone — no project dependencies)")
    print("=" * 74)

    print(f"\n  θ_relic = {math.degrees(THETA_RELIC):.4f}°  =  arctan(1/√8)")
    print(f"  αp/αs   = tan²θ = 1/8  =  {TAN2_THETA:.6f}")

    analytic_pwave_summary()

    results = {}

    for label in TEST_BPS:
        bp     = BENCHMARKS[label]
        m_chi  = bp["m_chi"]
        alpha_s = bp["alpha_s"]

        print(f"\n{'━' * 74}")
        print(f"  BENCHMARK: {label}")
        print(f"{'━' * 74}")
        print(f"  m_χ = {m_chi:.3f} GeV,  αs = {alpha_s:.4e}")

        alpha_p, a_gev, b_gev, a_scalar_gev = sv_coefficients(m_chi, alpha_s)

        a_cm       = a_gev       * GEV2_TO_CM3S
        b_cm       = b_gev       * GEV2_TO_CM3S
        a_scalar_cm = a_scalar_gev * GEV2_TO_CM3S

        print(f"  αp  = αs/8 = {alpha_p:.4e}")
        print(f"  y_s = {math.sqrt(4*math.pi*alpha_s):.4f},  "
              f"y_p = {math.sqrt(4*math.pi*alpha_p):.4f},  "
              f"y   = {math.sqrt(4*math.pi*alpha_s)/math.cos(THETA_RELIC):.4f}")

        # ── σv decomposition ────────────────────────────────────────────
        print(f"\n  σv DECOMPOSITION (mφ ≪ mχ limit):")
        print(f"    a  (s-wave from αs·αp): {a_cm:.4e} cm³/s  = {a_cm/PLANCK_SV:.4f}·σ_Planck")
        print(f"    b  (p-wave from αp²):   {b_cm:.4e} cm³/s")
        print(f"    b/a =  {b_gev/a_gev:.5f}  = 3/16 = {3.0/16:.5f}  (check: {abs(b_gev/a_gev - 3/16) < 1e-9})")
        print(f"    a_scalar(θ=0°):         {a_scalar_cm:.4e} cm³/s")
        print(f"    a/a_scalar = {a_gev/a_scalar_gev:.6f}  ← must = 1.0 at θ_relic")

        # ── Naive Ωh² (PI-8 formula, s-wave only) ───────────────────────
        omega_naive = omega_h2_naive(m_chi, alpha_s)

        # ── Freeze-out ──────────────────────────────────────────────────
        xf_s  = find_xf(m_chi, a_gev, 0.0)       # s-wave only
        xf_sp = find_xf(m_chi, a_gev, b_gev)     # s + p wave

        T_fo_MeV = m_chi * 1e3 / xf_sp

        print(f"\n  FREEZE-OUT:")
        print(f"    xf (s-wave only):  {xf_s:.3f}")
        print(f"    xf (s+p-wave):     {xf_sp:.3f}")
        print(f"    T_fo = mχ/xf =     {T_fo_MeV:.1f} MeV")
        print(f"    ⟨v²⟩_fo = 6/xf  = {6.0/xf_sp:.5f}")

        # ── σv at freeze-out ────────────────────────────────────────────
        sv_fo_s  = a_cm
        sv_fo_sp = a_cm + 6.0 * b_cm / xf_sp

        print(f"\n  σv AT FREEZE-OUT:")
        print(f"    s-wave only:  {sv_fo_s:.4e} cm³/s")
        print(f"    s + p-wave:   {sv_fo_sp:.4e} cm³/s")
        print(f"    p-wave boost: +{(sv_fo_sp/sv_fo_s - 1)*100:.2f}%")

        # ── Relic density ───────────────────────────────────────────────
        omega_s    = omega_h2(xf_s,  a_gev, 0.0)
        omega_sp   = omega_h2(xf_sp, a_gev, b_gev)
        D_needed   = omega_naive / OMEGA_TARGET
        D_achieved = omega_sp / OMEGA_TARGET

        print(f"\n  Ωh²:")
        print(f"    naive (s-wave, PI-8):  {omega_naive:.4f}  (D needed = {D_needed:.2f}×)")
        print(f"    s-wave (full xf):      {omega_s:.4f}")
        print(f"    s+p-wave (θ=19.47°):   {omega_sp:.4f}")
        print(f"    target:                {OMEGA_TARGET:.4f}")
        print(f"    still off by:          {D_achieved:.2f}×")
        print(f"    improvement from PI-8: {(omega_naive - omega_sp)/omega_naive*100:.2f}%")

        # ── What αp would be needed? ─────────────────────────────────────
        # Need: Ωh² = 0.12 → a_eff = PLANCK_SV / (Ωh² = 0.12) × 0.12 = PLANCK_SV
        # More precisely: a_eff_needed = PLANCK_SV (= 3e-26 cm³/s)
        # a_cm + 3b_cm/xf = PLANCK_SV
        # b_cm_needed = (PLANCK_SV - a_cm) × xf / 3
        b_cm_needed = (PLANCK_SV - a_cm) * xf_sp / 3.0
        if b_cm_needed > 0:
            b_gev_needed    = b_cm_needed / GEV2_TO_CM3S
            alpha_p_needed  = math.sqrt(b_gev_needed * m_chi**2 / (3.0 * math.pi))
            tan2_needed     = alpha_p_needed / alpha_s
        else:
            alpha_p_needed  = 0.0
            tan2_needed     = 0.0

        print(f"\n  TO REACH Ωh²=0.12 (via p-wave boost alone):")
        print(f"    Need b = {b_cm_needed:.3e} cm³/s")
        print(f"    Need αp = {alpha_p_needed:.4e}  (current = {alpha_p:.4e})")
        print(f"    Need αp/αs = tan²θ = {tan2_needed:.4f}  (current = {TAN2_THETA:.4f} = 1/8)")
        if tan2_needed > 0:
            print(f"    Need θ = {math.degrees(math.atan(math.sqrt(tan2_needed))):.1f}°  "
                  f"(current = {math.degrees(THETA_RELIC):.1f}°)")
            print(f"    Factor αp must increase: ×{alpha_p_needed/alpha_p:.0f}")

        results[label] = {
            "omega_naive": omega_naive,
            "omega_sp":    omega_sp,
            "xf":          xf_sp,
            "boost_pct":   (sv_fo_sp / sv_fo_s - 1) * 100,
            "D_needed":    D_needed,
            "D_achieved":  D_achieved,
        }

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n\n{'═' * 74}")
    print("  SUMMARY TABLE")
    print(f"{'═' * 74}")
    print(f"  {'BP':>8}  {'Ωh²(s-only)':>12}  {'Ωh²(s+p)':>10}  "
          f"{'p-boost':>9}  {'D_needed':>9}  {'D_got':>8}  {'PASS?':>6}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*6}")

    for label in TEST_BPS:
        r = results[label]
        ok = abs(r["omega_sp"] - OMEGA_TARGET) / OMEGA_TARGET < 0.10
        print(f"  {label:>8}  {r['omega_naive']:>12.4f}  {r['omega_sp']:>10.4f}  "
              f"{r['boost_pct']:>8.2f}%  {r['D_needed']:>9.2f}  "
              f"{r['D_achieved']:>8.2f}  {'✅' if ok else '✗':>6}")

    print(f"\n  Target: Ωh² = {OMEGA_TARGET}")

    print(f"""
  PHYSICAL EXPLANATION:
  ─────────────────────
  At θ_relic the pseudoscalar coupling is αp = αs/8. Therefore:

      b/a = 3αp² / (αs·αp·2) = 3αp/(2αs) = 3/16 ≈ 0.19  (universal)

  At freeze-out ⟨v²⟩ ≈ 0.27, the p-wave boosts σv by only:

      δσv/σv = 6b/(a·xf) = 6×(3/16)/22 ≈ 5%

  The factor of 2.94 needed for MAP requires a boost of +194%.
  The p-wave can only deliver +5%.  Missing by a factor of ~38×.

  Root cause: tan²(θ_relic) = 1/8 was chosen to reproduce the
  correct relic at θ=0 (s-wave). It was NOT chosen to maximize
  p-wave.  The angle that maximizes p-wave would be θ=90° (pure
  pseudoscalar), but that kills SIDM (αs=0).

  The T-breaking angle is constrained from both sides:
    θ small → SIDM works, relic overclosed (PI-8/PI-9 problem)
    θ large → SIDM weakens, CMB reionization constraints tighten

  CONCLUSION: PI-9 NEGATIVE.  p-wave at θ_relic corrects Ωh² by ~5%.
              The relic tension (factor ~3) survives T-breaking.
""")
    print(f"{'═' * 74}")
    print("  NEXT STEP: find benchmark region where mχ is large enough that")
    print("  Ωh²(s-wave) ≤ 0.12 already — i.e., mχ > mχ_crit(α).")
    print(f"{'═' * 74}")


if __name__ == "__main__":
    main()
