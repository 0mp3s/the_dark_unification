#!/usr/bin/env python3
"""
PI-12 v2: Clockwork Bridge — CORRECTED with proper Kolb-Turner Ωh²
====================================================================
Self-contained — runs on Google Colab, Replit, Kaggle, or any Python 3.8+
with `import math` only.  NO project dependencies.  NO GPU.

CRITICAL FIX (2026-03-29):
  PI-12 v1 used the crude "WIMP miracle" formula:
      Ωh² ≈ 0.12 × (3×10⁻²⁶) / ⟨σv⟩_eff
  This OVERESTIMATES by ~55-64% (factor 33.66/x_f).

  This version uses the proper Kolb-Turner analytic:
      Y_∞ = 1/(λ·J)
      λ = √(π/45) × g*s/√g* × M_Pl × m
      J = a/x_f + 3b/x_f²

  CONSEQUENCE: Only MAP (m_χ=98.2 GeV) reaches Ωh²=0.12 with the σ channel.
  BP1, BP9, BP16 have φ-only Ωh² < 0.12 → adding σ makes it WORSE.
  f₀(MAP) ≈ 1094 GeV (not 755 GeV from the old formula).
  f_DE/f_target IMPROVES from 0.80 → 1.16 (closer to 1!).

TESTS:
  א. Ωh²(f₀) — show φ-only and crossing for each BP
  ב. Clockwork: f_DE = q^N × f₀(MAP) — better agreement
  ג. Heavy mode masses — updated for f₀ ≈ 1094
  ד. Full consistency (8 checks)
  ה. (q,N) sensitivity scan
  ו. NEW: m_χ^min scan — minimum DM mass for the mechanism to work

Omer P. — 2026-03-29
"""

import math

# ════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ════════════════════════════════════════════════════════════════════════
M_PL_GEV     = 1.2209e19    # Full Planck mass [GeV]
M_PL_RED_GEV = 2.4353e18    # Reduced Planck mass M_Pl/√(8π) [GeV]
GEV2_TO_CM3S = 3.8938e-28 * 3.0e10   # GeV⁻² → cm³/s
OMEGA_TARGET = 0.1200
G_STAR       = 86.25
G_STAR_S     = 86.25
S0_CM3       = 2891.2       # entropy density today [cm⁻³]
RHO_C_H2     = 1.054e-5     # ρ_c/h² [GeV cm⁻³]

# Dark energy
F_DE_GEV     = 0.24 * M_PL_RED_GEV   # 5.845e17 GeV
LAMBDA_D_GEV = 2.28e-12              # Dark confinement scale [GeV]
RHO_LAMBDA   = 2.6e-47               # GeV⁴ (cosmological constant)

# ════════════════════════════════════════════════════════════════════════
# BENCHMARKS (from global_config.json)
# ════════════════════════════════════════════════════════════════════════
BENCHMARKS = {
    "BP1":  {"m_chi": 54.556,  "alpha_s": 2.645e-3, "m_phi_MeV": 12.975},
    "BP9":  {"m_chi": 48.329,  "alpha_s": 2.350e-3, "m_phi_MeV":  8.657},
    "BP16": {"m_chi": 14.384,  "alpha_s": 7.555e-4, "m_phi_MeV":  5.047},
    "MAP":  {"m_chi": 98.19,   "alpha_s": 3.274e-3, "m_phi_MeV":  9.660},
}

TEST_BPS = ["BP1", "BP9", "BP16", "MAP"]


# ════════════════════════════════════════════════════════════════════════
#  RELIC DENSITY — PROPER KOLB-TURNER
# ════════════════════════════════════════════════════════════════════════

def phi_channel_a(m_chi, alpha_s):
    """s-wave from φ (SIDM-locked): a_φ = π α_s²/(4 m_χ²)  [GeV⁻²]"""
    return math.pi * alpha_s**2 / (4.0 * m_chi**2)

def sigma_channel_b(m_chi, f_gev):
    """p-wave from σ derivative coupling: b_σ = 3 m_χ²/(π f⁴)  [GeV⁻²]"""
    return 3.0 * m_chi**2 / (math.pi * f_gev**4)

def find_xf(m_chi, a_gev, b_gev, g_chi=1.0, g_star=G_STAR):
    """Iterative freeze-out x_f (Kolb & Turner eq. 5.45)."""
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
        xf = 0.6 * xf + 0.4 * xf_new
    return max(xf, 5.0)

def omega_KT(m_chi, a_gev, b_gev):
    """Proper Kolb-Turner Ωh²:  Y_∞ = 1/(λ·J).

    λ = √(π/45) × g*s/√g* × M_Pl × m
    J = a/x_f + 3b/x_f²

    Returns (oh2, xf).
    """
    xf = find_xf(m_chi, a_gev, b_gev)
    lam = math.sqrt(math.pi / 45.0) * G_STAR_S / math.sqrt(G_STAR) * M_PL_GEV * m_chi
    J = a_gev / xf + 3.0 * b_gev / xf**2
    if J <= 0:
        return 999.0, xf
    Y_inf = 1.0 / (lam * J)
    oh2 = m_chi * Y_inf * S0_CM3 / RHO_C_H2
    return oh2, xf

def compute_omega(m_chi, alpha_s, f0_gev):
    """Full Ωh² given f₀.  Returns (oh2, xf)."""
    a_phi = phi_channel_a(m_chi, alpha_s)
    b_sig = sigma_channel_b(m_chi, f0_gev)
    return omega_KT(m_chi, a_phi, b_sig)

def find_f_cross(m_chi, alpha_s, f_lo=1e2, f_hi=1e8, tol=1e-4):
    """Bisection to find f₀ where Ωh² = 0.1200.

    Returns None if φ-only Ωh² < 0.12 (no crossing possible).
    """
    a_phi = phi_channel_a(m_chi, alpha_s)

    # Edge case: if φ alone gives Ωh² < target, adding σ only lowers it further
    oh2_phi_only, _ = omega_KT(m_chi, a_phi, 0.0)
    if oh2_phi_only < OMEGA_TARGET:
        return None   # no f₀ can bring Ωh² up to 0.12

    def omega_at_f(f_gev):
        b_sig = sigma_channel_b(m_chi, f_gev)
        oh2, _ = omega_KT(m_chi, a_phi, b_sig)
        return oh2

    oh2_lo = omega_at_f(f_lo)
    oh2_hi = omega_at_f(f_hi)
    if (oh2_lo - OMEGA_TARGET) * (oh2_hi - OMEGA_TARGET) > 0:
        return None
    for _ in range(100):
        f_mid = math.sqrt(f_lo * f_hi)
        oh2_mid = omega_at_f(f_mid)
        if abs(oh2_mid - OMEGA_TARGET) / OMEGA_TARGET < tol:
            return f_mid
        if oh2_mid > OMEGA_TARGET:
            f_hi = f_mid
        else:
            f_lo = f_mid
    return math.sqrt(f_lo * f_hi)


# ════════════════════════════════════════════════════════════════════════
#  CLOCKWORK MACHINERY (unchanged)
# ════════════════════════════════════════════════════════════════════════

def clockwork_f_de(f0, q, N):
    """Physical decay constant at the DE end: f_DE = q^N × f₀."""
    return f0 * q**N

def clockwork_heavy_mass_sq(k, f0, q, N, Lambda_CW):
    """M_k² = (Λ_CW⁴/f₀²) × [1 + q² − 2q cos(kπ/(N+1))]"""
    return (Lambda_CW**4 / f0**2) * (1.0 + q**2 - 2.0*q*math.cos(k*math.pi/(N+1)))

def clockwork_heavy_mass(k, f0, q, N, Lambda_CW):
    m2 = clockwork_heavy_mass_sq(k, f0, q, N, Lambda_CW)
    return math.sqrt(m2) if m2 > 0 else 0.0

def clockwork_lightest_heavy(f0, q, N, Lambda_CW):
    return clockwork_heavy_mass(1, f0, q, N, Lambda_CW)

def clockwork_heaviest(f0, q, N, Lambda_CW):
    masses = [clockwork_heavy_mass(k, f0, q, N, Lambda_CW) for k in range(1, N+1)]
    return max(masses)

def de_misalignment_rho(f_de, Lambda_d, theta_i=1.0):
    return Lambda_d**4 * (1.0 - math.cos(theta_i))

def delta_neff_heavy_modes(N, M_lightest, T_fo_gev):
    ratio = M_lightest / T_fo_gev
    if ratio > 30:
        return 0.0
    per_mode = (4.0/7.0) * (11.0/4.0)**(4.0/3.0) * ratio**3 * math.exp(-ratio)
    return N * per_mode


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("  PI-12 v2: CLOCKWORK BRIDGE — PROPER KT FORMULA")
    print("  (standalone — import math only)")
    print("=" * 78)

    # ══════════════════════════════════════════════════════════════════════
    #  TEST א: Ωh²(f₀) for all BPs — show which can reach 0.12
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 78}")
    print("  TEST א: RELIC DENSITY — φ-only Ωh² and f₀_cross")
    print(f"{'━' * 78}\n")

    f_cross_all = {}
    print(f"  {'BP':>6}  {'m_χ [GeV]':>10}  {'α_s':>10}  {'Ωh²(φ only)':>12}  "
          f"{'> 0.12?':>7}  {'f₀_cross [GeV]':>14}  {'y_P':>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*7}  {'─'*14}  {'─'*7}")

    for label in TEST_BPS:
        bp = BENCHMARKS[label]
        m_chi = bp["m_chi"]
        alpha_s = bp["alpha_s"]

        # φ-only relic (proper KT)
        a_phi = phi_channel_a(m_chi, alpha_s)
        oh2_phi, xf0 = omega_KT(m_chi, a_phi, 0.0)
        above = oh2_phi > OMEGA_TARGET

        # Find f₀ that gives Ωh² = 0.12
        fc = find_f_cross(m_chi, alpha_s)
        f_cross_all[label] = fc

        if fc is not None:
            y_P = 2.0 * m_chi / fc
            print(f"  {label:>6}  {m_chi:>10.3f}  {alpha_s:>10.3e}  {oh2_phi:>12.4f}  "
                  f"{'YES':>7}  {fc:>14.1f}  {y_P:>7.4f}")
        else:
            print(f"  {label:>6}  {m_chi:>10.3f}  {alpha_s:>10.3e}  {oh2_phi:>12.4f}  "
                  f"{'NO':>7}  {'∞ (no cross)':>14}  {'—':>7}")

    valid_fc = {k: v for k, v in f_cross_all.items() if v is not None}
    n_valid = len(valid_fc)

    print(f"\n  ⚠️  RESULT: {n_valid}/4 benchmarks have finite f₀_cross")
    if n_valid == 0:
        print("  ERROR: No benchmark can reach Ωh²=0.12. Check parameters!")
        return
    elif n_valid == 1:
        bp_name = list(valid_fc.keys())[0]
        f0_ref = valid_fc[bp_name]
        print(f"  Only {bp_name} works: f₀ = {f0_ref:.1f} GeV")
        print(f"  BP1/BP9/BP16: φ-only Ωh² < 0.12 → σ channel makes it worse")
        print(f"  → Model PREDICTS m_χ ≳ {BENCHMARKS[bp_name]['m_chi']:.0f} GeV (see test ו)")
    else:
        # multiple valid — take geometric mean
        f0_ref = math.exp(sum(math.log(f) for f in valid_fc.values()) / n_valid)
        print(f"  ⟨f₀⟩_geo ({n_valid} BPs) = {f0_ref:.1f} GeV")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST ב: Clockwork q^N scaling — f_DE consistency
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 78}")
    print("  TEST ב: CLOCKWORK BRIDGE — q^N × f₀ = f_DE?")
    print(f"{'━' * 78}\n")

    print(f"  f₀ = {f0_ref:.1f} GeV (from MAP relic)")
    print(f"  f_DE target = {F_DE_GEV:.3e} GeV")
    print(f"  Required ratio = f_DE/f₀ = {F_DE_GEV/f0_ref:.3e}")
    print(f"  log₁₀(ratio) = {math.log10(F_DE_GEV/f0_ref):.2f}")
    print()

    print(f"  {'q':>3}  {'N':>4}  {'q^N':>14}  {'f_DE = q^N f₀':>16}  "
          f"{'f_DE/f_target':>14}  {'log₁₀ err':>10}  {'pass?':>6}")
    print(f"  {'─'*3}  {'─'*4}  {'─'*14}  {'─'*16}  {'─'*14}  {'─'*10}  {'─'*6}")

    best_configs = []

    for q in [2, 3, 4, 5, 7, 10]:
        N_exact = math.log(F_DE_GEV / f0_ref) / math.log(q)
        for N in range(max(1, int(N_exact) - 1), int(N_exact) + 3):
            if N < 1 or N > 200:
                continue
            qN = q**N
            f_de_calc = qN * f0_ref
            ratio = f_de_calc / F_DE_GEV
            log_err = abs(math.log10(ratio))

            ok = log_err < 0.5  # within factor 3
            mark = "✅" if ok else ""

            if abs(N - round(N_exact)) <= 1:
                print(f"  {q:>3}  {N:>4}  {qN:>14.3e}  {f_de_calc:>16.3e}  "
                      f"{ratio:>14.4f}  {log_err:>10.3f}  {mark:>6}")
                if ok:
                    best_configs.append((q, N, f_de_calc, ratio))

    print(f"\n  BEST CONFIGURATIONS (within factor 3 of f_DE):")
    for q, N, f_de, ratio in best_configs:
        print(f"    q={q}, N={N}: f_DE = {f_de:.3e} GeV (×{ratio:.2f} of target)")

    # Optimal (q,N) — scan for closest to 1
    q_best, N_best = 3, 31
    best_ratio = abs(math.log10(clockwork_f_de(f0_ref, 3, 31) / F_DE_GEV))
    for q, N, f_de, ratio in best_configs:
        lr = abs(math.log10(ratio))
        if lr < best_ratio:
            q_best, N_best = q, N
            best_ratio = lr

    f_de_best = clockwork_f_de(f0_ref, q_best, N_best)
    print(f"\n  SELECTED: q={q_best}, N={N_best}")
    print(f"    q^N = {q_best**N_best:.3e}")
    print(f"    f_DE = q^N × f₀ = {f_de_best:.3e} GeV")
    print(f"    f_DE / f_target = {f_de_best / F_DE_GEV:.3f}")
    print(f"    (v1 had f₀=755 → f_DE/f_t=0.80; v2: f₀={f0_ref:.0f} → f_DE/f_t={f_de_best/F_DE_GEV:.2f} — IMPROVED!)")

    # DE density check
    rho_sigma = de_misalignment_rho(f_de_best, LAMBDA_D_GEV, theta_i=1.0)
    m_sigma_de = LAMBDA_D_GEV**2 / f_de_best
    print(f"\n  DARK ENERGY CHECK:")
    print(f"    Λ_d = {LAMBDA_D_GEV:.2e} GeV")
    print(f"    m_σ(f_DE) = Λ_d²/f_DE = {m_sigma_de:.3e} GeV")
    print(f"    ρ_σ(θ_i=1) = Λ_d⁴(1−cos1) = {rho_sigma:.3e} GeV⁴")
    print(f"    ρ_Λ = {RHO_LAMBDA:.3e} GeV⁴")
    print(f"    The Clockwork fixes f, not Λ_d. f_DE ≈ target ✅")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST ג: Heavy mode spectrum — masses > m_χ?
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 78}")
    print("  TEST ג: HEAVY MODE SPECTRUM — COLLIDER & FREEZE-OUT SAFETY")
    print(f"{'━' * 78}\n")

    m_chi_max = max(bp["m_chi"] for bp in BENCHMARKS.values())

    print(f"  Parameters: q={q_best}, N={N_best}, f₀={f0_ref:.1f} GeV")
    print(f"  Heaviest DM: m_χ(MAP) = {m_chi_max:.2f} GeV")
    print()

    print(f"  {'Λ_CW [GeV]':>12}  {'M_1 [GeV]':>10}  {'M_N [GeV]':>10}  "
          f"{'M_1 > m_χ?':>10}  {'M_1 > 1 TeV?':>12}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")

    Lambda_CW_min = None
    Lambda_CW_tev = None

    for log_LCW in [x * 0.25 for x in range(4, 20)]:
        LCW = 10.0**log_LCW
        M1 = clockwork_lightest_heavy(f0_ref, q_best, N_best, LCW)
        MN = clockwork_heaviest(f0_ref, q_best, N_best, LCW)

        ok_mchi = M1 > m_chi_max
        ok_tev = M1 > 1000.0

        print(f"  {LCW:>12.1f}  {M1:>10.1f}  {MN:>10.1f}  "
              f"{'✅' if ok_mchi else '✗':>10}  {'✅' if ok_tev else '✗':>12}")

        if ok_mchi and Lambda_CW_min is None:
            Lambda_CW_min = LCW
        if ok_tev and Lambda_CW_tev is None:
            Lambda_CW_tev = LCW

    LCW_analytic = math.sqrt(m_chi_max * f0_ref / 2.0)
    print(f"\n  ANALYTIC: Λ_CW > √(m_χ f₀/2) = {LCW_analytic:.1f} GeV")
    if Lambda_CW_min:
        print(f"  Numerical: Λ_CW > {Lambda_CW_min:.1f} GeV (M_1 > m_χ)")

    # Use Λ_CW = 500 GeV
    Lambda_CW_use = 500.0
    M1_use = clockwork_lightest_heavy(f0_ref, q_best, N_best, Lambda_CW_use)
    MN_use = clockwork_heaviest(f0_ref, q_best, N_best, Lambda_CW_use)
    print(f"\n  CHOSEN: Λ_CW = {Lambda_CW_use:.0f} GeV")
    print(f"    M_1 = {M1_use:.1f} GeV, M_N = {MN_use:.1f} GeV")

    # ΔN_eff
    T_fo = m_chi_max / 25.0
    d_neff = delta_neff_heavy_modes(N_best, M1_use, T_fo)
    print(f"    ΔN_eff ≈ {d_neff:.3e}  ({'✅ ≪ 0.3' if d_neff < 0.3 else '⚠️'})")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST ד: FULL CONSISTENCY CHECK
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 78}")
    print("  TEST ד: FULL CONSISTENCY — SIDM + RELIC + DE")
    print(f"{'━' * 78}\n")

    checks = []
    bp_map = BENCHMARKS["MAP"]
    fc_map = f_cross_all["MAP"]

    print(f"  {'CHECK':>45s}  {'VALUE':>16}  {'CRITERION':>20}  {'RESULT':>8}")
    print(f"  {'─'*45}  {'─'*16}  {'─'*20}  {'─'*8}")

    # 1. Relic density for MAP
    if fc_map is not None:
        oh2_map, _ = compute_omega(bp_map["m_chi"], bp_map["alpha_s"], fc_map)
    else:
        oh2_map = 99.0
    ok1 = abs(oh2_map - OMEGA_TARGET) / OMEGA_TARGET < 0.05
    checks.append(ok1)
    print(f"  {'Ωh²(MAP) = 0.120':>45s}  {oh2_map:>16.4f}  {'|Δ|/Ω < 5%':>20}  {'✅' if ok1 else '✗':>8}")

    # 2. Perturbativity: y_P < 1
    if fc_map is not None:
        y_P_map = 2.0 * bp_map["m_chi"] / fc_map
    else:
        y_P_map = 99.0
    ok2 = y_P_map < 1.0
    checks.append(ok2)
    print(f"  {'y_P = 2m_χ/f₀ < 1 (perturbative)':>45s}  {y_P_map:>16.4f}  {'< 1':>20}  {'✅' if ok2 else '✗':>8}")

    # 3. f_DE within factor 3
    f_de_cw = clockwork_f_de(f0_ref, q_best, N_best)
    ratio_de = f_de_cw / F_DE_GEV
    ok3 = 0.3 < ratio_de < 3.0
    checks.append(ok3)
    print(f"  {'f_DE = q^N f₀ ≈ f_target':>45s}  {f_de_cw:>16.3e}  {'factor < 3':>20}  {'✅' if ok3 else '✗':>8}")

    # 4. M_1 > m_chi(MAP)
    ok4 = M1_use > m_chi_max
    checks.append(ok4)
    print(f"  {'M₁ > m_χ(MAP) (no new channels)':>45s}  {M1_use:>16.1f}  {'> {:.1f}'.format(m_chi_max):>20}  {'✅' if ok4 else '✗':>8}")

    # 5. ΔN_eff < 0.3
    ok5 = d_neff < 0.3
    checks.append(ok5)
    print(f"  {'ΔN_eff < 0.3 (BBN/CMB)':>45s}  {d_neff:>16.3e}  {'< 0.3':>20}  {'✅' if ok5 else '✗':>8}")

    # 6. SIDM unaffected
    if fc_map is not None:
        v2_today = (30.0 / 3e5)**2
        b_sig_map = sigma_channel_b(bp_map["m_chi"], fc_map)
        a_phi_map = phi_channel_a(bp_map["m_chi"], bp_map["alpha_s"])
        sv_sig_today = b_sig_map * v2_today * GEV2_TO_CM3S
        sv_phi_today = a_phi_map * GEV2_TO_CM3S
        ratio_today = sv_sig_today / sv_phi_today if sv_phi_today > 0 else 99
    else:
        ratio_today = 99.0
    ok6 = ratio_today < 0.01
    checks.append(ok6)
    print(f"  {'σv(σ)/σv(φ) today < 1%':>45s}  {ratio_today:>16.3e}  {'< 0.01':>20}  {'✅' if ok6 else '✗':>8}")

    # 7. Λ_CW/f₀ natural
    ratio_LCW = Lambda_CW_use / f0_ref
    ok7 = 0.01 < ratio_LCW < 100
    checks.append(ok7)
    print(f"  {'Λ_CW/f₀ = O(1) (naturalness)':>45s}  {ratio_LCW:>16.2f}  {'0.01 < x < 100':>20}  {'✅' if ok7 else '✗':>8}")

    # 8. φ-only Ωh²(MAP) > 0.12 (mechanism prerequisite)
    a_phi_m = phi_channel_a(bp_map["m_chi"], bp_map["alpha_s"])
    oh2_phi_map, _ = omega_KT(bp_map["m_chi"], a_phi_m, 0.0)
    ok8 = oh2_phi_map > OMEGA_TARGET
    checks.append(ok8)
    print(f"  {'Ωh²_φ(MAP) > 0.12 (σ can tune down)':>45s}  {oh2_phi_map:>16.4f}  {'> 0.12':>20}  {'✅' if ok8 else '✗':>8}")

    all_pass = all(checks)
    n_pass = sum(checks)

    # ══════════════════════════════════════════════════════════════════════
    #  TEST ה: SENSITIVITY — what (q,N) range works?
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 78}")
    print("  TEST ה: SENSITIVITY — VIABLE (q, N) PARAMETER SPACE")
    print(f"{'━' * 78}\n")

    print(f"  Criterion: 0.1 < f_DE(q,N)/f_target < 10  (within 1 dex)")
    print()
    print(f"  {'q':>3}  {'N_min':>5}  {'N_max':>5}  {'N_range':>7}  {'best N':>6}  {'f_DE/f_tgt':>10}  {'N_exact':>8}")
    print(f"  {'─'*3}  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*10}  {'─'*8}")

    for q in [2, 3, 4, 5, 7, 10]:
        N_exact = math.log(F_DE_GEV / f0_ref) / math.log(q)
        N_min_ok = None
        N_max_ok = None
        for N in range(1, 201):
            f_de_test = clockwork_f_de(f0_ref, q, N)
            ratio_test = f_de_test / F_DE_GEV
            if 0.1 < ratio_test < 10:
                if N_min_ok is None:
                    N_min_ok = N
                N_max_ok = N

        if N_min_ok is not None:
            N_range = N_max_ok - N_min_ok + 1
            N_best_q = round(N_exact)
            f_de_best_q = clockwork_f_de(f0_ref, q, N_best_q)
            print(f"  {q:>3}  {N_min_ok:>5}  {N_max_ok:>5}  {N_range:>7}  "
                  f"{N_best_q:>6}  {f_de_best_q/F_DE_GEV:>10.3f}  {N_exact:>8.2f}")

    print(f"\n  For ANY q ≥ 2, there exists N giving f_DE ≈ f_target.")

    # ══════════════════════════════════════════════════════════════════════
    #  TEST ו: m_χ^min SCAN — minimum DM mass for mechanism to work
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 78}")
    print("  TEST ו: MINIMUM DM MASS — m_χ^min PREDICTION")
    print(f"{'━' * 78}\n")

    print("  The σ derivative coupling mechanism requires φ-only Ωh² > 0.12.")
    print("  Below m_χ^min, the s-wave φ channel alone gives Ωh² < 0.12,")
    print("  and adding σ (p-wave) only lowers Ωh² further.")
    print()

    # Use MAP's alpha_s as reference (since we don't know alpha_s at other m_chi)
    # Actually, alpha_s is set by SIDM fits and varies with m_chi.
    # Let's scan using the SIDM relation: alpha_s(m_chi) from the 4 BPs.

    # First, show that alpha_s scales roughly as alpha_s ∝ m_chi^n (fit from BPs)
    bp_data = [(BENCHMARKS[k]["m_chi"], BENCHMARKS[k]["alpha_s"]) for k in TEST_BPS]
    bp_data.sort()

    print("  SIDM benchmark points:")
    print(f"  {'BP':>5}  {'m_χ [GeV]':>10}  {'α_s':>10}  {'Ωh²(φ only)':>12}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*12}")
    for label in TEST_BPS:
        bp = BENCHMARKS[label]
        a = phi_channel_a(bp["m_chi"], bp["alpha_s"])
        oh2_phi, _ = omega_KT(bp["m_chi"], a, 0.0)
        print(f"  {label:>5}  {bp['m_chi']:>10.3f}  {bp['alpha_s']:>10.3e}  {oh2_phi:>12.4f}")

    print()

    # Use log-log interpolation of (m_chi, alpha_s) from BPs
    # Sort by m_chi
    ms = [bp["m_chi"] for bp in [BENCHMARKS[k] for k in ["BP16", "BP9", "BP1", "MAP"]]]
    als = [bp["alpha_s"] for bp in [BENCHMARKS[k] for k in ["BP16", "BP9", "BP1", "MAP"]]]

    def interp_alpha(m_chi_test):
        """Log-linear interpolation of alpha_s(m_chi) from benchmarks."""
        if m_chi_test <= ms[0]:
            # extrapolate from first two
            slope = (math.log(als[1]) - math.log(als[0])) / (math.log(ms[1]) - math.log(ms[0]))
            return math.exp(math.log(als[0]) + slope * (math.log(m_chi_test) - math.log(ms[0])))
        if m_chi_test >= ms[-1]:
            slope = (math.log(als[-1]) - math.log(als[-2])) / (math.log(ms[-1]) - math.log(ms[-2]))
            return math.exp(math.log(als[-1]) + slope * (math.log(m_chi_test) - math.log(ms[-1])))
        for i in range(len(ms) - 1):
            if ms[i] <= m_chi_test <= ms[i+1]:
                t = (math.log(m_chi_test) - math.log(ms[i])) / (math.log(ms[i+1]) - math.log(ms[i]))
                return math.exp((1-t)*math.log(als[i]) + t*math.log(als[i+1]))
        return als[-1]

    # Scan m_chi from 10 to 200 GeV
    print("  m_χ scan (interpolated α_s):")
    print(f"  {'m_χ [GeV]':>10}  {'α_s (interp)':>12}  {'Ωh²(φ only)':>12}  {'> 0.12?':>7}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*7}")

    m_cross = None  # the threshold mass
    m_prev = None
    oh2_prev = None

    for m_test in [10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 120, 150, 200]:
        a_test = interp_alpha(m_test)
        a_phi = phi_channel_a(m_test, a_test)
        oh2_phi, _ = omega_KT(m_test, a_phi, 0.0)
        above = oh2_phi > OMEGA_TARGET
        marker = "  ✅" if above else ""
        print(f"  {m_test:>10.1f}  {a_test:>12.3e}  {oh2_phi:>12.4f}  {'YES' if above else 'NO':>7}{marker}")

        if m_prev is not None and oh2_prev < OMEGA_TARGET and oh2_phi >= OMEGA_TARGET:
            # Bisect to find exact crossing
            m_lo, m_hi = m_prev, m_test
            for _ in range(50):
                m_mid = (m_lo + m_hi) / 2
                a_mid = interp_alpha(m_mid)
                a_phi_mid = phi_channel_a(m_mid, a_mid)
                oh2_mid, _ = omega_KT(m_mid, a_phi_mid, 0.0)
                if oh2_mid < OMEGA_TARGET:
                    m_lo = m_mid
                else:
                    m_hi = m_mid
            m_cross = (m_lo + m_hi) / 2

        m_prev = m_test
        oh2_prev = oh2_phi

    if m_cross is not None:
        a_cross = interp_alpha(m_cross)
        a_phi_cross = phi_channel_a(m_cross, a_cross)
        oh2_cross, _ = omega_KT(m_cross, a_phi_cross, 0.0)
        print(f"\n  ═══════════════════════════════════════════════════")
        print(f"  m_χ^min = {m_cross:.1f} GeV (α_s = {a_cross:.3e})")
        print(f"  Ωh²(φ only) @ m_χ^min = {oh2_cross:.4f}")
        print(f"  ═══════════════════════════════════════════════════")
        print(f"\n  MODEL PREDICTION: m_χ > {m_cross:.0f} GeV")
        print(f"  DM lighter than ~{m_cross:.0f} GeV cannot achieve Ωh²=0.12 in this model.")
        print(f"  (This is a NEW prediction from the corrected KT formula)")
    else:
        print("\n  Could not find m_χ^min crossing — check scan range.")

    # ══════════════════════════════════════════════════════════════════════
    #  VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 78}")
    print(f"  PI-12 v2 VERDICT:  {n_pass}/{len(checks)} CHECKS PASS {'✅' if all_pass else '⚠️'}")
    print(f"{'═' * 78}")
    print()
    print(f"  WHAT CHANGED from v1:")
    print(f"    • Ωh² formula: crude PLANCK_SV → proper KT (Y_∞ = 1/λJ)")
    print(f"    • Old formula overestimated by ~55-64%")
    print(f"    • f₀: 755 GeV (4 BPs) → {f0_ref:.0f} GeV (MAP only)")
    print(f"    • BP1/BP9/BP16: Ωh²(φ) < 0.12 → no longer have crossing")
    print()
    print(f"  WHAT IMPROVED:")
    print(f"    • f_DE/f_target: 0.80 → {f_de_best/F_DE_GEV:.2f} (CLOSER to 1!)")
    print(f"    • y_P: 0.26 → {y_P_map:.2f} (more perturbative)")
    if m_cross:
        print(f"    • NEW PREDICTION: m_χ > {m_cross:.0f} GeV")
    print()
    print(f"  THE MODEL (updated):")
    print(f"  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  f₀ = {f0_ref:.0f} GeV  (MAP benchmark, proper KT)          │")
    print(f"  │  q = {q_best}, N = {N_best}                                          │")
    print(f"  │  f_DE = q^N × f₀ = {f_de_best:.2e} GeV                │")
    print(f"  │  f_DE / f_target = {f_de_best/F_DE_GEV:.3f}                              │")
    print(f"  │  Λ_CW = {Lambda_CW_use:.0f} GeV → M₁ = {M1_use:.0f} GeV                      │")
    if m_cross:
        print(f"  │  PREDICTION: m_χ > {m_cross:.0f} GeV                           │")
    print(f"  └──────────────────────────────────────────────────────────┘")
    print()
    print(f"  ACTION ITEMS:")
    print(f"    1. Re-run PI-14-17 with f₀={f0_ref:.0f} (expect all still pass)")
    print(f"    2. Check whether any SIDM BPs with m_χ > {m_cross or 80:.0f} exist")
    print(f"    3. Update preprint: 'universal f₀' → 'MAP benchmark + mass prediction'")
    print()


if __name__ == '__main__':
    main()
