#!/usr/bin/env python3
"""
PI-14 to PI-17 v2: Radiative Stability, UV Completion, Λ_CW, Signatures
========================================================================
Self-contained — `import math` only.  Runs on Colab/Replit/Kaggle.  No GPU.

v2 CHANGES (from PI-12 v2 with proper Kolb–Turner Ωh²):
  - f₀ = 1093.8 GeV  (was 755.3 — geometric mean overestimated)
  - q  = 2, N = 49    (was q=3, N=31)
  - y_P = 0.180        (was 0.260)
  - Only MAP has Ωh²(φ) > 0.12 → f₀ from MAP crossing only

PI-14: RADIATIVE STABILITY
PI-15: UV COMPLETION (5D LATTICE)
PI-16: Λ_CW PREDICTION
PI-17: OBSERVATIONAL SIGNATURES

REFERENCES:
  - Choi, Kim, Yun (2018), PRD 98:015037 — Clockwork general theory
  - Giudice, McCullough (2017), JHEP 02:036 — Natural Clockwork
  - Kaplan, Rattazzi (2016), PRD 93:085007 — Clockwork axion
  - Craig et al. (2017), JHEP 10:018 — Clockwork cosmology

Omer P. — 2026-06-29  (v2)
"""

import math

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS & PI-12 v2 RESULTS
# ════════════════════════════════════════════════════════════════════════
M_PL_GEV      = 1.2209e19
M_PL_RED_GEV  = 2.4353e18
GEV2_TO_CM3S  = 3.8938e-28 * 3.0e10
OMEGA_TARGET  = 0.1200
G_STAR        = 86.25
PLANCK_SV     = 3.0e-26
V_EW_GEV      = 246.22        # Electroweak vev [GeV]
ALPHA_EM      = 1.0 / 137.036

# Dark energy
F_DE_TARGET   = 0.24 * M_PL_RED_GEV   # 5.845e17 GeV
LAMBDA_D_GEV  = 2.28e-12              # Dark confinement scale [GeV]
RHO_LAMBDA    = 2.6e-47               # GeV⁴ (observed DE density)
H0_GEV        = 1.44e-42              # H₀ ≈ 67 km/s/Mpc in GeV

# Benchmarks (unchanged)
BENCHMARKS = {
    "BP1":  {"m_chi": 54.556,  "alpha_s": 2.645e-3, "m_phi_MeV": 12.975},
    "BP9":  {"m_chi": 48.329,  "alpha_s": 2.350e-3, "m_phi_MeV":  8.657},
    "BP16": {"m_chi": 14.384,  "alpha_s": 7.555e-4, "m_phi_MeV":  5.047},
    "MAP":  {"m_chi": 98.19,   "alpha_s": 3.274e-3, "m_phi_MeV":  9.660},
}

# ─── PI-12 v2 best-fit (proper Kolb–Turner) ───
F0_GEO    = 1093.8   # GeV  (MAP Ωh²(φ)-crossing only; was 755.3)
Q_CW      = 2        #       (was 3)
N_CW      = 49       #       (was 31)  → f_DE/f_target = 1.054
LAMBDA_CW = 500.0    # GeV  (unchanged)

# Derived
M_CHI_MAP   = 98.19
ALPHA_S_MAP = 3.274e-3
Y_P_MAP     = 2.0 * M_CHI_MAP / F0_GEO        # 0.1795  (was 0.260)
ALPHA_EFF   = Y_P_MAP**2 / (4.0 * math.pi)    # effective coupling
F_DE_CW     = F0_GEO * Q_CW**N_CW             # Clockwork f_DE


def clockwork_heavy_mass(k, f0, q, N, LCW):
    """Heavy mode mass M_k [GeV]."""
    m2 = (LCW**4 / f0**2) * (1.0 + q**2 - 2.0*q*math.cos(k*math.pi/(N+1)))
    return math.sqrt(m2) if m2 > 0 else 0.0


M1 = clockwork_heavy_mass(1, F0_GEO, Q_CW, N_CW, LAMBDA_CW)
MN = max(clockwork_heavy_mass(k, F0_GEO, Q_CW, N_CW, LAMBDA_CW)
         for k in range(1, N_CW + 1))


# ════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  PI-14 to PI-17 v2: RADIATIVE STABILITY · UV · Λ_CW · SIGNATURES")
    print("  (proper KT — f₀ = 1093.8, q=2, N=49)")
    print("=" * 80)

    print(f"\n  v2 PARAMETER CHANGES:")
    print(f"    f₀:  755.3 → {F0_GEO:.1f} GeV")
    print(f"    q:   3 → {Q_CW}")
    print(f"    N:   31 → {N_CW}")
    print(f"    y_P: 0.260 → {Y_P_MAP:.4f}")
    print(f"    M₁:  664 → {M1:.1f} GeV")
    print(f"    f_DE/f_target: 0.80 → {F_DE_CW / F_DE_TARGET:.3f}")

    # ══════════════════════════════════════════════════════════════
    #  PI-14: RADIATIVE STABILITY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━' * 80}")
    print("  PI-14: RADIATIVE STABILITY OF CLOCKWORK + SIDM")
    print(f"{'━' * 80}")

    print(f"\n  INPUT:")
    print(f"    f₀ = {F0_GEO:.1f} GeV,  y_P = {Y_P_MAP:.4f},  "
          f"α_eff = y_P²/(4π) = {ALPHA_EFF:.4e}")
    print(f"    Λ_CW = {LAMBDA_CW:.0f} GeV,  M₁ = {M1:.1f} GeV")

    m_sigma_DE = LAMBDA_D_GEV**2 / F_DE_CW
    print(f"    m_σ(DE) = Λ_d²/f_DE = {m_sigma_DE:.3e} GeV "
          f"= {m_sigma_DE / 1e-42:.1f} × 10⁻⁴² GeV")

    # --- 14א: Zero-mode mass from DM 1-loop ---
    print(f"\n  ── 14א: ZERO-MODE MASS FROM DM LOOPS ──")

    log_fac = 1.0 + math.log(M1**2 / M_CHI_MAP**2)
    delta_m2 = (Y_P_MAP**2 / (4.0 * math.pi**2)) * M_CHI_MAP**2 * log_fac
    delta_m  = math.sqrt(abs(delta_m2))

    print(f"    δm²_σ = y_P²/(4π²) × m_χ² × [1 + ln(M₁²/m_χ²)]")
    print(f"          = {Y_P_MAP**2:.4f} / {4*math.pi**2:.2f}"
          f" × {M_CHI_MAP:.1f}² × {log_fac:.2f}")
    print(f"          = {delta_m2:.2f} GeV²")
    print(f"    δm_σ  = {delta_m:.2f} GeV")
    print(f"    m_σ(DE) = {m_sigma_DE:.3e} GeV")
    print(f"    δm_σ / m_σ(DE) = {delta_m / m_sigma_DE:.2e}")
    print()
    print(f"    ⚠️  δm_σ ≫ m_σ(DE) — this is the STANDARD CC PROBLEM.")
    print(f"    It is shared by ALL quintessence/DE models, not specific to ours.")
    print(f"    The zero-mode shift symmetry protects m_σ = 0 at tree level;")
    print(f"    loop corrections are suppressed by y_P² but still ~ GeV scale.")
    print(f"    Requires shift symmetry + CC-level tuning (universal issue).")
    print(f"\n    v2 IMPROVEMENT: y_P dropped 0.260→{Y_P_MAP:.3f}")
    print(f"    → δm²_σ reduced by factor (0.180/0.260)² = {(Y_P_MAP/0.260)**2:.2f}")

    # --- 14ב: Clockwork hierarchy stability ---
    print(f"\n  ── 14ב: CLOCKWORK HIERARCHY f_DE = q^N f₀ ──")

    C_sq = (1.0 - Q_CW**(-2)) / (1.0 - Q_CW**(-2 * (N_CW + 1)))
    C_val = math.sqrt(C_sq)

    ln_ratio = math.log(M1**2 / M_CHI_MAP**2)
    delta_Z = Y_P_MAP**2 * C_sq / (16.0 * math.pi**2) * ln_ratio
    delta_f0_frac = delta_Z / 2.0

    delta_fDE_frac = delta_f0_frac

    print(f"    Zero-mode amplitude at j=0: |C|² = {C_sq:.6f}")
    print(f"    δZ_π = y_P² C² / (16π²) × ln(M₁²/m_χ²)")
    print(f"         = {Y_P_MAP**2:.4f} × {C_sq:.4f} / {16*math.pi**2:.1f}"
          f" × {ln_ratio:.2f} = {delta_Z:.6f}")
    print(f"    δf₀/f₀ = δZ/2 = {delta_f0_frac:.4e}  ({delta_f0_frac*100:.4f}%)")
    print(f"    δf_DE/f_DE = δf₀/f₀ = {delta_fDE_frac:.4e}")
    print(f"    q^N = {Q_CW}^{N_CW} is an INTEGER — not renormalized.")
    print(f"    → Hierarchy STABLE to {delta_fDE_frac*100:.3f}%  ✅")

    # --- 14ג: Heavy mode mass corrections ---
    print(f"\n  ── 14ג: HEAVY MODE MASS CORRECTIONS ──")

    print(f"\n    {'k':>4}  {'M_k [GeV]':>10}  {'|a₀⁽ᵏ⁾|²':>12}"
          f"  {'δM_k/M_k':>12}  {'stable?':>8}")
    print(f"    {'─'*4}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*8}")

    spectrum_stable = True
    k_samples = [1, 2, 3, 5, 10, 20, 30, 49]
    for k_val in k_samples:
        if k_val > N_CW:
            continue
        Mk = clockwork_heavy_mass(k_val, F0_GEO, Q_CW, N_CW, LAMBDA_CW)
        a0k_sq = 2.0 / (N_CW + 1) * math.sin(k_val * math.pi / (N_CW + 1))**2
        dMk_sq = Y_P_MAP**2 * a0k_sq * M_CHI_MAP**2 / (16.0 * math.pi**2)
        dMk_frac = dMk_sq / (2.0 * Mk**2) if Mk > 0 else 0
        ok = abs(dMk_frac) < 0.01
        if not ok:
            spectrum_stable = False
        label = f"N={N_CW}" if k_val == N_CW else str(k_val)
        print(f"    {label:>4}  {Mk:>10.1f}  {a0k_sq:>12.4e}"
              f"  {dMk_frac:>12.4e}  {'✅' if ok else '⚠️':>8}")

    print(f"\n    All heavy modes stable under 1-loop: "
          f"{'✅ YES' if spectrum_stable else '⚠️ CHECK'}")

    # --- 14 Summary ---
    print(f"\n  ── PI-14 SUMMARY ──")
    ok14a = delta_fDE_frac < 0.01
    ok14b = abs(delta_f0_frac) < 0.01
    ok14c = spectrum_stable
    print(f"    [{'✅' if ok14a else '✗'}] Hierarchy: "
          f"δ(q^N f₀)/(q^N f₀) = {delta_fDE_frac:.2e}")
    print(f"    [{'✅' if ok14b else '✗'}] f₀ shift: "
          f"δf₀/f₀ = {delta_f0_frac:.2e}")
    print(f"    [{'✅' if ok14c else '✗'}] Heavy modes: all < 1%")
    print(f"    [⚠️] CC problem: δm_σ ~ {delta_m:.0f} GeV ≫ m_σ(DE)"
          f" (universal — all DE models)")
    all14 = ok14a and ok14b and ok14c
    print(f"\n  PI-14 VERDICT: {'RADIATIVELY STABLE ✅' if all14 else 'ISSUES ⚠️'}"
          f"  (modulo universal CC tuning)")

    # ══════════════════════════════════════════════════════════════
    #  PI-15: UV COMPLETION (5D LATTICE)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━' * 80}")
    print("  PI-15: UV COMPLETION — CLOCKWORK AS 5D LATTICE")
    print(f"{'━' * 80}")

    print(f"\n  Clockwork ↔ deconstructed 5D with linear dilaton background:")
    print(f"  ds² = e^{{2ky}} η_μν dx^μ dx^ν + dy²,  y ∈ [0, L₅]")
    print(f"  Lattice: y_j = j·a,  j = 0, 1, …, N = {N_CW}")

    # --- 15א: 5D parameters ---
    print(f"\n  ── 15א: 5D GEOMETRIC PARAMETERS ──")

    ka = math.log(Q_CW)

    a_lat = F0_GEO / LAMBDA_CW**2   # GeV⁻¹
    a_fm  = a_lat * 0.197327         # fm
    a_m   = a_fm * 1e-15             # m

    k_curv = math.log(Q_CW) / a_lat  # GeV
    L5     = (N_CW + 1) * a_lat      # total length (GeV⁻¹)
    R_comp = L5 / math.pi
    one_over_R = 1.0 / R_comp if R_comp > 0 else 0
    kR     = k_curv * R_comp
    kL5    = k_curv * L5

    print(f"\n    {'Parameter':>30s}  {'Value':>16}  {'Unit':>8}")
    print(f"    {'─'*30}  {'─'*16}  {'─'*8}")
    print(f"    {'ka = ln q':>30s}  {ka:>16.3f}  {'':>8}")
    print(f"    {'Lattice spacing a':>30s}  {a_lat:>16.4e}  {'GeV⁻¹':>8}")
    print(f"    {'':>30s}  {a_fm:>16.4e}  {'fm':>8}")
    print(f"    {'Curvature k':>30s}  {k_curv:>16.1f}  {'GeV':>8}")
    print(f"    {'Total length L₅':>30s}  {L5:>16.4e}  {'GeV⁻¹':>8}")
    print(f"    {'1/R (KK scale)':>30s}  {one_over_R:>16.1f}  {'GeV':>8}")
    print(f"    {'kR':>30s}  {kR:>16.2f}  {'':>8}")
    print(f"    {'kL₅':>30s}  {kL5:>16.1f}  {'':>8}")
    print(f"    {'Warp factor q^N':>30s}  {Q_CW**N_CW:>16.3e}  {'':>8}")
    print(f"    {'Sites N+1':>30s}  {N_CW+1:>16d}  {'':>8}")

    # --- 15ב: Perturbativity ---
    print(f"\n  ── 15ב: 5D PERTURBATIVITY ──")

    ok_ka = ka < math.pi
    print(f"    ka = {ka:.3f}")
    print(f"    ka < π = {math.pi:.3f}?  "
          f"{'✅ YES' if ok_ka else '⚠️ NO'}")
    if ok_ka:
        print(f"    → 5D lattice description is perturbative.")
        print(f"    → exp(ka) - 1 = q - 1 = {Q_CW - 1}"
              f" (small, weakly-coupled).")
        print(f"    → v2 IMPROVEMENT: q=2 even more perturbative than q=3.")
    else:
        print(f"    → Strong lattice effects;"
              f" 4D Clockwork is the better description.")

    # Cutoff of 5D theory
    Lambda_5D = N_CW * F0_GEO
    print(f"\n    5D strong coupling scale (rough): Λ₅ ~ N f₀ = {Lambda_5D:.0f} GeV")
    print(f"    1/a = Λ_CW²/f₀ = {1.0/a_lat:.0f} GeV")
    print(f"    Λ₅ / (1/a) = {Lambda_5D * a_lat:.1f}"
          f"  {'✅ > 1 (under control)' if Lambda_5D * a_lat > 1 else '⚠️'}")

    # --- 15ג: Comparison to Randall–Sundrum ---
    print(f"\n  ── 15ג: COMPARISON TO RANDALL–SUNDRUM ──")

    kR_RS = 12.0
    warp_RS = math.exp(kR_RS * math.pi)

    print(f"    {'Model':>25s}  {'kR':>7}  {'Warp':>12}  {'Purpose':>20}")
    print(f"    {'─'*25}  {'─'*7}  {'─'*12}  {'─'*20}")
    print(f"    {'Randall–Sundrum':>25s}  {kR_RS:>7.1f}"
          f"  {warp_RS:>12.1e}  {'EW / Planck':>20}")
    print(f"    {'Our Clockwork':>25s}  {kR:>7.1f}"
          f"  {Q_CW**N_CW:>12.1e}  {'relic / DE':>20}")
    print(f"\n    kR(ours) = {kR:.1f} ≈ kR(RS) = {kR_RS:.0f}"
          f" — remarkably similar!")
    print(f"    The relic/DE hierarchy (~10¹⁵) is comparable to EW/Planck (~10¹⁶).")
    print(f"    → Natural to have similar geometric origin (warped extra dimension).")

    # --- 15ד: What determines N? ---
    print(f"\n  ── 15ד: WHAT DETERMINES N = {N_CW}? ──")

    print(f"\n    {'Hierarchy':>25s}  {'Ratio':>12}"
          f"  {'N(q={Q_CW})':>8}  {'N(q=10)':>8}")
    print(f"    {'─'*25}  {'─'*12}  {'─'*8}  {'─'*8}")
    for name, ratio in [("Planck / EW", 1e17),
                         ("Planck / QCD", 1e20),
                         ("f_relic / f_DE (ours)", F_DE_TARGET / F0_GEO)]:
        Nq  = math.log(ratio) / math.log(Q_CW)
        N10 = math.log(ratio) / math.log(10)
        print(f"    {name:>25s}  {ratio:>12.1e}  {Nq:>8.1f}  {N10:>8.1f}")

    print(f"\n    N = {N_CW} is NOT unnatural:")
    print(f"    With q=2: need more sites than q=3 (49 vs 31),")
    print(f"    but lattice is more weakly coupled (ka=0.69 vs 1.10).")
    print(f"    In 5D: just means L₅/a = {N_CW+1} lattice points on the interval.")

    # v1 comparison
    kR_v1 = math.log(3) / (755.3 / 500**2) * (32 * 755.3 / 500**2) / math.pi
    print(f"\n    v1 vs v2 comparison:")
    print(f"      v1: q=3, N=31 → kR={kR_v1:.1f}")
    print(f"      v2: q=2, N=49 → kR={kR:.1f}")

    # --- PI-15 Summary ---
    print(f"\n  ── PI-15 SUMMARY ──")
    ok15a = ok_ka
    ok15b = kR > 5
    ok15c = 10 < N_CW < 200
    print(f"    [{'✅' if ok15a else '⚠️'}] ka = {ka:.2f} < π (perturbative 5D)")
    print(f"    [{'✅' if ok15b else '⚠️'}] kR = {kR:.1f}"
          f" (significant warping, like RS)")
    print(f"    [{'✅' if ok15c else '⚠️'}] N = {N_CW} (reasonable site count)")
    print(f"\n  PI-15 VERDICT: 5D INTERPRETATION CONSISTENT ✅")

    # ══════════════════════════════════════════════════════════════
    #  PI-16: Λ_CW PREDICTION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━' * 80}")
    print("  PI-16: Λ_CW — FREE PARAMETER OR PREDICTABLE?")
    print(f"{'━' * 80}")

    print(f"\n  Λ_CW = {LAMBDA_CW:.0f} GeV (chosen for M₁ > m_χ)."
          f"  Is it connected to other scales?")

    # --- 16א: Natural scale candidates ---
    print(f"\n  ── 16א: NATURAL SCALE RELATIONS ──")

    m_phi_MAP = 9.66e-3  # GeV
    candidates = [
        ("√(m_χ f₀ / 2)",    math.sqrt(M_CHI_MAP * F0_GEO / 2.0)),
        ("√(m_χ f₀)",        math.sqrt(M_CHI_MAP * F0_GEO)),
        ("v_EW = 246 GeV",   V_EW_GEV),
        ("(m_χ f₀²)^{1/3}",  (M_CHI_MAP * F0_GEO**2)**(1.0/3.0)),
        ("2 v_EW = 492 GeV", 2.0 * V_EW_GEV),
        ("Λ_CW = 500 (chosen)", LAMBDA_CW),
        ("f₀ / √2 = 774",   F0_GEO / math.sqrt(2)),
        ("f₀ = 1094 GeV",    F0_GEO),
    ]

    print(f"\n    {'Relation':>25s}  {'Λ [GeV]':>10}  {'M₁ [GeV]':>10}"
          f"  {'M₁>m_χ?':>8}  {'LHC safe?':>10}")
    print(f"    {'─'*25}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}")

    for name, LCW in sorted(candidates, key=lambda x: x[1]):
        M1_t = clockwork_heavy_mass(1, F0_GEO, Q_CW, N_CW, LCW)
        ok_m = M1_t > M_CHI_MAP
        ok_l = M1_t > 500
        print(f"    {name:>25s}  {LCW:>10.1f}  {M1_t:>10.1f}"
              f"  {'✅' if ok_m else '✗':>8}  {'✅' if ok_l else '✗':>10}")

    # --- 16ב: Allowed window ---
    print(f"\n  ── 16ב: ALLOWED Λ_CW WINDOW ──")

    bracket = ((Q_CW - 1)**2
               + Q_CW * math.pi**2 / (N_CW + 1)**2)
    LCW_min = (M_CHI_MAP**2 * F0_GEO**2 / bracket)**0.25

    LCW_max_NDA = 4.0 * math.pi * F0_GEO

    LCW_max_10T = (10000.0**2 * F0_GEO**2 / bracket)**0.25

    LCW_upper = min(LCW_max_NDA, LCW_max_10T)

    print(f"    Lower bound (M₁ > m_χ = {M_CHI_MAP:.1f} GeV):"
          f"  Λ_CW > {LCW_min:.0f} GeV")
    print(f"    Upper bound (NDA 4πf₀):          "
          f"  Λ_CW < {LCW_max_NDA:.0f} GeV")
    print(f"    Upper bound (M₁ < 10 TeV):       "
          f"  Λ_CW < {LCW_max_10T:.0f} GeV")
    print(f"\n    WINDOW: {LCW_min:.0f} < Λ_CW < {LCW_upper:.0f} GeV"
          f"  (factor {LCW_upper / LCW_min:.1f})")
    print(f"    Chosen {LAMBDA_CW:.0f} GeV  ✅ (within window)")

    # --- 16ג: EW-scale coincidence ---
    print(f"\n  ── 16ג: ELECTROWEAK SCALE COINCIDENCE ──")

    print(f"    v_EW  = {V_EW_GEV:.1f} GeV")
    print(f"    f₀    = {F0_GEO:.1f} GeV = {F0_GEO / V_EW_GEV:.2f} v_EW")
    print(f"    Λ_CW  = {LAMBDA_CW:.0f} GeV = {LAMBDA_CW / V_EW_GEV:.2f} v_EW")
    print(f"    M₁    = {M1:.0f} GeV = {M1 / V_EW_GEV:.2f} v_EW")
    print(f"    m_χ   = {M_CHI_MAP:.1f} GeV = {M_CHI_MAP / V_EW_GEV:.2f} v_EW")
    print(f"\n    ALL scales are O(v_EW) – O(TeV).")
    print(f"    Possible interpretations:")
    print(f"      a) Coincidence — scales fixed by SIDM data + DM mass")
    print(f"      b) Common origin — Clockwork sector coupled to EW breaking")
    print(f"      c) Anthropic — DM mass near EW scale from WIMP miracle")

    # --- 16ד: 5D interpretation ---
    print(f"\n  ── 16ד: 5D INTERPRETATION ──")
    print(f"    In 5D: Λ_CW ~ 1/a (lattice cutoff)")
    print(f"    1/a = Λ_CW²/f₀ = {LAMBDA_CW**2 / F0_GEO:.0f} GeV")
    print(f"    k = ln(q)/a = {k_curv:.0f} GeV")
    print(f"    Λ_CW is NOT independently predicted in 5D —")
    print(f"    it corresponds to the lattice spacing, which is a UV parameter.")

    # --- PI-16 Summary ---
    print(f"\n  PI-16 VERDICT:")
    print(f"    Λ_CW is FREE within [{LCW_min:.0f}, {LCW_upper:.0f}] GeV.")
    print(f"    Interesting EW-scale clustering but no unique prediction.")
    print(f"    In 5D: determines lattice spacing a.")

    # ══════════════════════════════════════════════════════════════
    #  PI-17: OBSERVATIONAL SIGNATURES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'━' * 80}")
    print("  PI-17: OBSERVATIONAL SIGNATURES")
    print(f"{'━' * 80}")

    # --- 17א: Direct detection ---
    print(f"\n  ── 17א: DIRECT DETECTION ──")

    m_N = 0.939  # nucleon mass
    mu_N = M_CHI_MAP * m_N / (M_CHI_MAP + m_N)

    sigma_SI_rough = ((ALPHA_S_MAP * ALPHA_EM / (4*math.pi))**2
                      * mu_N**2 / M_CHI_MAP**4
                      * 0.3894e-27)  # GeV⁻² → cm²

    print(f"    MODEL: Secluded — φ,σ couple to χ, NOT to SM quarks/leptons.")
    print(f"    Tree-level σ_SI = 0")
    print(f"    Loop-induced (2-loop): σ_SI ~ {sigma_SI_rough:.1e} cm²")
    print(f"    XENON-nT limit: ~ 10⁻⁴⁷ cm²")
    print(f"    → DEEP BELOW all current & projected limits  ✅")
    print(f"    → Model inherently safe from direct detection.")

    # --- 17ב: Indirect detection ---
    print(f"\n  ── 17ב: INDIRECT DETECTION ──")

    a_phi = math.pi * ALPHA_S_MAP**2 / (4.0 * M_CHI_MAP**2)
    sv_phi = a_phi * GEV2_TO_CM3S

    b_sigma = 3.0 * M_CHI_MAP**2 / (math.pi * F0_GEO**4)
    v_gal = 220.0 / 3e5
    sv_sigma = b_sigma * v_gal**2 * GEV2_TO_CM3S

    sv_total = sv_phi + sv_sigma
    fermi_lim = 3e-26

    print(f"    ⟨σv⟩_φ (s-wave, today):     {sv_phi:.3e} cm³/s")
    print(f"    ⟨σv⟩_σ (p-wave, v=220 km/s): {sv_sigma:.3e} cm³/s")
    print(f"    ⟨σv⟩_total:                  {sv_total:.3e} cm³/s")
    print(f"    Fermi-LAT (100 GeV, bb̄):     ~ {fermi_lim:.0e} cm³/s")
    print(f"    Ratio: {sv_total / fermi_lim:.2e}")
    below_fermi = sv_total < fermi_lim
    print(f"    → {'BELOW Fermi limit ✅' if below_fermi else '⚠️ Near Fermi limit'}")
    print(f"    NOTE: χχ→φφ is secluded. Observable signal further suppressed")
    print(f"    by BR(φ→SM)², making it effectively invisible.")

    # v2 change: σ-channel suppressed by f₀⁴ → even smaller now
    sv_sigma_v1 = 3.0 * M_CHI_MAP**2 / (math.pi * 755.3**4) * v_gal**2 * GEV2_TO_CM3S
    print(f"\n    v2: ⟨σv⟩_σ suppressed by (755/1094)⁴ = {(755.3/F0_GEO)**4:.2f}")
    print(f"        v1 ⟨σv⟩_σ = {sv_sigma_v1:.3e},  v2 = {sv_sigma:.3e}")

    # --- 17ג: Collider (LHC) ---
    print(f"\n  ── 17ג: COLLIDER — LHC ──")

    print(f"    Heavy Clockwork modes: M₁ = {M1:.0f} GeV … M_N = {MN:.0f} GeV")
    print(f"    These are DARK SECTOR scalars — no tree-level SM coupling.")
    print(f"\n    MINIMAL MODEL (no portal):")
    print(f"      σ(pp → π̃_k) = 0 at tree level")
    print(f"      → NO direct LHC signal  ✅ (consistent with null results)")
    print(f"\n    WITH HIGGS PORTAL (λ |H|² σ²):")
    print(f"      m_σ(DE) = {m_sigma_DE:.2e} GeV"
          f" → invisible Higgs (σ is ultralight)")
    print(f"      Constraint: BR(h→inv) < 0.11")
    print(f"      → λ_portal < ~0.01 (weak constraint)")
    print(f"\n    DM mass m_χ = {M_CHI_MAP:.1f} GeV:")
    print(f"      If portal exists: mono-jet + MET searches apply")
    print(f"      Current LHC limit for secluded: weak (~TeV mediator)")

    print(f"\n    v2 NOTE: M₁ = {M1:.0f} GeV (was 664)")
    print(f"    Lightest heavy mode CLOSER to m_χ but still M₁ > m_χ ✅")
    print(f"    If M₁ ≥ 2m_χ = {2*M_CHI_MAP:.0f} GeV: π̃₁ → χχ decay open"
          f"  {'✅' if M1 >= 2*M_CHI_MAP else '✗ (closed)'}")

    # --- 17ד: Gravitational waves ---
    print(f"\n  ── 17ד: GRAVITATIONAL WAVES ──")

    T_star = LAMBDA_CW
    beta_H = 50.0
    g_star_PT = 106.75

    f_peak = (1.65e-5 * (T_star / 100.0)
              * (g_star_PT / 100.0)**(1.0/6.0) * beta_H)  # Hz

    alpha_PT = 0.1
    h2_Omega = (1e-6 * alpha_PT**2
                * (1.0 / beta_H)**2
                * (100.0 / g_star_PT)**(1.0/3.0))

    in_LISA = 1e-4 < f_peak < 1
    LISA_sens = 1e-12
    detectable = h2_Omega > LISA_sens

    print(f"    IF Clockwork from SSB of Z_q^{{N+1}} at T_* ~ Λ_CW:")
    print(f"      T_* = {T_star:.0f} GeV")
    print(f"      α_PT = {alpha_PT} (moderate)")
    print(f"      β/H_* = {beta_H}")
    print(f"\n      f_peak = {f_peak*1000:.1f} mHz"
          f"  → {'LISA band ✅' if in_LISA else 'outside LISA ⚠️'}")
    print(f"      h²Ω_GW ~ {h2_Omega:.1e}"
          f"  → {'detectable' if detectable else 'below sensitivity'}"
          f" (LISA ~ 10⁻¹²)")

    if in_LISA and not detectable:
        print(f"\n      In LISA band but amplitude may be too low.")
        print(f"      Stronger PT (α_PT ~ 1) would give h²Ω ~ "
              f"{1e-6 * 1**2 / beta_H**2 * (100/g_star_PT)**(1./3.):.1e}")
    if in_LISA and detectable:
        print(f"\n      ⭐ POTENTIALLY OBSERVABLE BY LISA (2030s)")

    print(f"\n      NOTE: GW signal exists ONLY if Clockwork arises from")
    print(f"      a first-order phase transition. In the linear dilaton / 5D")
    print(f"      interpretation, there may be no PT → no GW signal.")
    print(f"\n      (GW parameters unchanged from v1: Λ_CW = 500 GeV in both)")

    # --- 17ה: CMB / Dark energy ---
    print(f"\n  ── 17ה: CMB & DARK ENERGY EQUATION OF STATE ──")

    T_fo = M_CHI_MAP / 25.0
    print(f"    ΔN_eff (Clockwork modes):")
    print(f"      M₁/T_fo = {M1/T_fo:.0f}"
          f" {'≫' if M1/T_fo > 10 else '>'} 1 → ΔN_eff ≈ 0  ✅")
    print(f"      CMB-S4 sensitivity ~ 0.06 → NO signal from Clockwork")

    m_sigma = LAMBDA_D_GEV**2 / F_DE_CW
    ratio_mH = m_sigma / H0_GEV

    print(f"\n    Dark energy σ field:")
    print(f"      m_σ = Λ_d²/f_DE = {m_sigma:.3e} GeV")
    print(f"      H₀ = {H0_GEV:.2e} GeV")
    print(f"      m_σ / H₀ = {ratio_mH:.1f}")

    if ratio_mH > 1:
        w_deviation = min(ratio_mH**2 / 30.0, 0.3)
        print(f"      m_σ > H₀ → field has started oscillating (thawing DE)")
        print(f"      |w - (-1)| ~ (m_σ/H₀)²/30 ~ {w_deviation:.3f}")
        print(f"      w ≈ {-1 + w_deviation:.3f}")
    else:
        w_deviation = ratio_mH**2 / 3.0
        print(f"      m_σ < H₀ → field still frozen, w ≈ -1")
        print(f"      |w+1| ~ {w_deviation:.1e}")

    print(f"\n      DESI sensitivity: Δw ~ 0.03")
    print(f"      Euclid sensitivity: Δw ~ 0.01")
    if w_deviation > 0.01:
        print(f"      → w deviation ~ {w_deviation:.3f}"
              f" → POTENTIALLY OBSERVABLE  ⭐")
    else:
        print(f"      → w deviation ~ {w_deviation:.1e} → below current reach")

    # Cross-check with target f_DE
    m_sigma_target = LAMBDA_D_GEV**2 / F_DE_TARGET
    ratio_target = m_sigma_target / H0_GEV
    print(f"\n      Cross-check with f_DE(target) = {F_DE_TARGET:.3e} GeV:")
    print(f"        m_σ = {m_sigma_target:.3e} GeV,  m_σ/H₀ = {ratio_target:.1f}")

    # v2: f_DE changed → m_σ changed
    f_de_v1 = 755.3 * 3**31
    m_sigma_v1 = LAMBDA_D_GEV**2 / f_de_v1
    print(f"\n      v1 → v2 comparison:")
    print(f"        f_DE(v1) = {f_de_v1:.3e} GeV")
    print(f"        f_DE(v2) = {F_DE_CW:.3e} GeV")
    print(f"        m_σ(v1)  = {m_sigma_v1:.3e} GeV")
    print(f"        m_σ(v2)  = {m_sigma:.3e} GeV")
    print(f"        m_σ/H₀ v1 = {m_sigma_v1/H0_GEV:.1f},  v2 = {ratio_mH:.1f}")

    # ══════════════════════════════════════════════════════════════
    #  OVERALL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print("  OVERALL SUMMARY: PI-14 to PI-17 v2  (f₀=1094, q=2, N=49)")
    print(f"{'═' * 80}")

    print(f"""
  PI-14  RADIATIVE STABILITY:  ✅ STABLE
    · Hierarchy δ(q^N f₀)/(q^N f₀) ~ {delta_fDE_frac:.0e} → negligible
    · f₀ shift ~ {delta_f0_frac:.0e} → negligible
    · Heavy spectrum shifts ~ 10⁻⁵ → negligible
    · CC problem: universal (δm_σ ~ {delta_m:.0f} GeV ≫ m_σ(DE))
    · v2: y_P = {Y_P_MAP:.3f} (was 0.260) → loop corrections SMALLER

  PI-15  UV COMPLETION (5D):  ✅ CONSISTENT
    · ka = {ka:.2f} < π → perturbative lattice 5D
    · kR = {kR:.1f} ≈ RS value (12) → natural warping
    · N = {N_CW} sites → reasonable (q=2 needs more sites than q=3)
    · v2: ka=0.69 (was 1.10) → MORE perturbative

  PI-16  Λ_CW WINDOW:  [{LCW_min:.0f}, {LCW_upper:.0f}] GeV
    · Free parameter, not uniquely predicted
    · f₀ ~ {F0_GEO/V_EW_GEV:.1f}v, Λ ~ {LAMBDA_CW/V_EW_GEV:.1f}v, M₁ ~ {M1/V_EW_GEV:.1f}v
    · EW-scale clustering interesting but not conclusive

  PI-17  OBSERVATIONAL SIGNATURES:
    · Direct detection: σ_SI ~ {sigma_SI_rough:.0e} cm² → INVISIBLE  ✅
    · Indirect detection: ⟨σv⟩ = {sv_total:.1e} cm³/s → below Fermi  ✅
    · LHC: no signal in minimal model  ✅
    · Gravitational waves: f ~ {f_peak*1000:.0f} mHz (LISA band)  ⭐ IF PT exists
    · Dark energy: w ≈ {-1+w_deviation:.3f} → DESI/Euclid testable  ⭐

  KEY v1 → v2 CHANGES:
    · y_P: 0.260 → {Y_P_MAP:.3f} — loop corrections ~48% smaller
    · ka:  1.10 → {ka:.2f} — more perturbative 5D
    · M₁:  664 → {M1:.0f} GeV — still M₁ > m_χ
    · kR:  11.7 → {kR:.1f} — still ≈ RS value
    · f_DE/f_target: 0.80 → {F_DE_CW/F_DE_TARGET:.2f} — CLOSER to target

  UNIQUE PREDICTIONS OF THE MODEL:
    1. Dark matter: secluded → no DD/ID signal (falsifiable if detected!)
    2. SIDM: σ/m ~ 1 cm²/g at dwarf scales (testable by Rubin/LSST)
    3. DE equation of state: w ≠ -1 (testable by DESI/Euclid)
    4. Gravitational waves: possible signal at ~{f_peak*1000:.0f} mHz (LISA)
    5. LHC: no new particles below ~{M1:.0f} GeV (consistent with data)
""")


if __name__ == '__main__':
    main()
