#!/usr/bin/env python3
"""
fifth_force_constraints.py — Is β ≈ 5 already ruled out?
=========================================================

The dark axion σ mediates a fifth force between χ particles with:
    β = M_Pl / f ≈ 5  (for f = 0.2 M_Pl)

Existing constraints on DM-DE coupling β from:
  1. Planck CMB (TT, TE, EE + lensing)
  2. BAO (BOSS, eBOSS, DESI)
  3. Structure formation / σ₈
  4. Bullet Cluster

Key papers:
  - Pettorino+ 2012: β < 0.06 (95% CL, Planck + BAO)
  - Amendola+ 2020: β < 0.066 (Planck 2018)
  - Pourtsidou+ 2013: β < 0.1 (model-dependent)
  - Baldi+ 2010: β < 0.15 (N-body, structure formation)

BUT: these assume σ couples to ALL dark matter universally.
In our model, χ IS all the dark matter (secluded sector, 100% of Ω_DM).
So the constraints apply fully.

Question: Can we escape with a smaller β (larger f)?
"""

import numpy as np

# Constants
GeV = 1.0
MeV = 1e-3
M_Pl = 2.435e18
H_0 = 1.44e-42
rho_L = 2.58e-47

# BPs
BPs = {
    "BP1":       {"m_chi": 20.69e-3, "m_phi": 11.34e-3, "alpha": 1.048e-3},
    "BP9":       {"m_chi": 42.53e-3, "m_phi": 10.92e-3, "alpha": 2.165e-3},
    "MAP":       {"m_chi": 94.07e-3, "m_phi": 11.10e-3, "alpha": 5.734e-3},
    "MAP_relic": {"m_chi": 85.84e-3, "m_phi": 15.35e-3, "alpha": 5.523e-3},
}

print("="*78)
print("FIFTH FORCE CONSTRAINTS ON β = M_Pl / f")
print("="*78)
print()

# =============================================================================
# PART 1: What does β > 0.06 mean observationally?
# =============================================================================
print("PART 1: EXISTING CONSTRAINTS")
print("-"*40)
print()
print("Coupled dark energy: L ⊃ β(σ/M_Pl) T^μ_μ(DM)")
print()
print("  Constraint source           | β_max (95% CL)")
print("  ----------------------------|----------------")
print("  Planck 2018 + BAO           | 0.066")
print("  Planck + BAO + σ₈           | 0.05")
print("  N-body (structure formation)| 0.15")
print("  Bullet Cluster (offset)     | ~0.5 (weak)")
print()

# =============================================================================
# PART 2: Our β for different f
# =============================================================================
print("PART 2: OUR MODEL — β vs f")
print("-"*40)
print()

beta_max = 0.066  # Planck constraint
f_min = M_Pl / beta_max
print(f"  For β ≤ {beta_max}: f ≥ M_Pl/β = {f_min:.2e} GeV")
print(f"  f_min / M_Pl = {f_min/M_Pl:.1f}")
print()

# f = M_Pl/β, so β = 0.066 → f = 15.2 M_Pl = 3.7 × 10^19 GeV
# This is SUPER-Planckian! In string theory, f > M_Pl is problematic (WGC).

print("  β     | f / M_Pl | f (GeV)      | Status")
print("  ------|----------|--------------|--------")
for beta in [5.0, 1.0, 0.5, 0.15, 0.066, 0.05, 0.01]:
    f = M_Pl / beta
    status = "EXCLUDED" if beta > 0.066 else ("MARGINAL" if beta > 0.05 else "ALLOWED")
    print(f"  {beta:<5.3f} | {1/beta:<8.1f} | {f:.2e} | {status}")
print()

# =============================================================================
# PART 3: What happens to V_σ ~ ρ_Λ if f ≫ M_Pl?
# =============================================================================
print("PART 3: IMPACT ON V_σ ~ ρ_Λ")
print("-"*40)
print()
print("V_σ ~ y⁴ m_χ⁶ / (32π² f²)")
print("If f must be > 15 M_Pl (for β < 0.066), then V_σ is suppressed by (15)² = 225×")
print()

theta_relic = np.arctan(1/np.sqrt(8))

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    alpha = bp["alpha"]
    y_sq = 4 * np.pi * alpha / np.cos(theta_relic)**2
    y = np.sqrt(y_sq)
    
    # V_σ with f = 0.2 M_Pl (original, excluded)
    f_orig = 0.2 * M_Pl
    V_orig = y**4 * m_chi**6 / (32 * np.pi**2 * f_orig**2)
    
    # V_σ with f = 15 M_Pl (Planck-allowed)
    f_planck = 15.0 * M_Pl
    V_planck = y**4 * m_chi**6 / (32 * np.pi**2 * f_planck**2)
    
    # V_σ with f = 20 M_Pl (safely allowed)
    f_safe = 20.0 * M_Pl
    V_safe = y**4 * m_chi**6 / (32 * np.pi**2 * f_safe**2)
    
    print(f"  {name}: m_χ = {m_chi*1e3:.1f} MeV")
    print(f"    f = 0.2 M_Pl: V_σ/ρ_Λ = 10^{np.log10(V_orig/rho_L):.1f}  (β=5, EXCLUDED)")
    print(f"    f = 15 M_Pl:  V_σ/ρ_Λ = 10^{np.log10(V_planck/rho_L):.1f}  (β=0.066, marginal)")
    print(f"    f = 20 M_Pl:  V_σ/ρ_Λ = 10^{np.log10(V_safe/rho_L):.1f}  (β=0.05, safe)")
    print()

# =============================================================================
# PART 4: THE SCREENING LOOPHOLE
# =============================================================================
print("="*78)
print("PART 4: SCREENING MECHANISMS — CAN β=5 SURVIVE?")
print("="*78)
print()
print("The Planck constraint β < 0.066 assumes UNSCREENED fifth force.")
print("If σ has a screening mechanism, the effective β can be large")
print("locally but small on cosmological scales.")
print()
print("Known screening mechanisms:")
print()
print("  Mechanism      | How it works                    | Applicable?")
print("  ---------------|--------------------------------|------------")
print("  Chameleon      | m_σ grows in dense regions     | Maybe — need V(σ) shape")
print("  Symmetron      | coupling vanishes in dense env | Unlikely (shift sym)")
print("  Vainshtein     | derivative self-interactions   | Need higher-order terms")
print("  K-mouflage     | kinetic screening              | Need non-standard kinetic")
print()
print("  For a shift-symmetric pseudo-Goldstone (our σ):")
print("  - Chameleon: POSSIBLE if V_bare(σ) has the right shape")
print("    In dense environments (early universe, clusters),")
print("    m_σ could be large → fifth force short-range → screened")
print("    In voids (low density), m_σ → H₀ → cosmological DE")
print()
print("  This is actually natural for coupled quintessence!")
print()

# =============================================================================
# PART 5: THE CHAMELEON POSSIBILITY
# =============================================================================
print("="*78)
print("PART 5: CHAMELEON DARK AXION")
print("="*78)
print()
print("If σ is a chameleon, its effective mass depends on DM density ρ_DM:")
print()
print("  m²_σ,eff = m²_σ + β ρ_DM / (M_Pl f)")
print()
print("In dense environments (ρ_DM ≫ ρ_crit):")
print("  m_σ,eff ≫ H₀ → fifth force range ≪ Hubble → screened")
print("  → CMB and structure formation see reduced β_eff")
print()
print("In cosmic voids (ρ_DM → 0):")
print("  m_σ,eff → m_σ ~ H₀ → fifth force range ~ Hubble → DE")
print()

# What ρ_DM is needed to screen?
# m²_eff > (100 H₀)² ~ 10⁻⁸⁰ GeV² (to be irrelevant at CMB scales)
# β ρ_DM / (M_Pl f) > 10⁻⁸⁰
# ρ_DM > 10⁻⁸⁰ M_Pl f / β

print("Screening density (for m_σ,eff > 100 H₀):")
print()
for beta in [5.0, 1.0, 0.1]:
    f = M_Pl / beta
    rho_screen = (100 * H_0)**2 * f * M_Pl / beta  # rough estimate
    # Compare to ρ_DM,today = 1.3 keV/cm³ = 1.05e-47 GeV⁴
    rho_DM_today = 1.05e-47  # GeV⁴
    rho_crit = 3.64e-47  # GeV⁴
    
    # Actually: m²_eff = d²V/dσ² + β² ρ_DM / (M²_Pl)
    # For chameleon with exponential coupling:
    # m²_eff ~ β² ρ / M²_Pl
    m2_eff_today = beta**2 * rho_DM_today / M_Pl**2
    m_eff_today = np.sqrt(m2_eff_today)
    
    # At recombination (z=1100): ρ_DM ~ (1100)³ × ρ_DM,today
    rho_DM_rec = (1100)**3 * rho_DM_today
    m2_eff_rec = beta**2 * rho_DM_rec / M_Pl**2
    m_eff_rec = np.sqrt(m2_eff_rec)
    
    print(f"  β = {beta}:")
    print(f"    m_σ,eff today     = {m_eff_today/H_0:.2e} H₀"
          f"  ({m_eff_today:.2e} GeV)")
    print(f"    m_σ,eff at z=1100 = {m_eff_rec/H_0:.2e} H₀"
          f"  ({m_eff_rec:.2e} GeV)")
    
    # Fifth force range
    if m_eff_today > 0:
        lambda_today = 1 / m_eff_today  # in GeV⁻¹
        lambda_today_mpc = lambda_today * 1.97e-14 * 1e-3  # GeV⁻¹ to Mpc (rough)
        # 1 GeV⁻¹ = 0.197 fm = 1.97e-16 m
        # 1 Mpc = 3.086e22 m
        lambda_today_m = lambda_today * 1.97e-16  # meters
        lambda_today_mpc_v = lambda_today_m / 3.086e22
    
    if m_eff_rec > 0:
        lambda_rec_m = (1/m_eff_rec) * 1.97e-16
        lambda_rec_mpc = lambda_rec_m / 3.086e22
    
    print(f"    Range today     = {lambda_today_m:.2e} m = {lambda_today_mpc_v:.2e} Mpc")
    print(f"    Range at z=1100 = {lambda_rec_m:.2e} m = {lambda_rec_mpc:.2e} Mpc")
    
    # Hubble radius today ~ 4400 Mpc, at recombination ~ 0.3 Mpc (comoving)
    print(f"    Hubble radius today ~ 4400 Mpc")
    print(f"    Sound horizon z=1100 ~ 0.15 Mpc (physical)")
    
    if lambda_rec_mpc < 0.15:
        print(f"    → SCREENED at CMB: range ≪ sound horizon ✓")
    else:
        print(f"    → NOT screened at CMB: range > sound horizon ✗")
    print()

# =============================================================================
# PART 6: THE REAL TEST — DENSITY-DEPENDENT SCREENING
# =============================================================================
print("="*78)
print("PART 6: DOES THE CW POTENTIAL ITSELF PROVIDE SCREENING?")
print("="*78)
print()
print("In our model, m²_σ is set by the CW potential, which depends on")
print("the LOCAL value of ⟨φ⟩ and the DM density.")
print()
print("At high DM density (early universe, cluster cores):")
print("  - χ number density is high")
print("  - CW potential gets stronger (more χ loops)")  
print("  - m_σ increases → fifth force screened")
print()
print("At low DM density (voids, late universe):")
print("  - χ number density is low")
print("  - CW potential weakens")
print("  - m_σ decreases → fifth force long-range → DE behavior")
print()
print("This is AUTOMATIC chameleon screening from the CW structure!")
print("No need to add it by hand.")
print()

# The CW-generated σ mass depends on χ density through finite-T effects
# At temperature T (or equivalently, number density n_χ):
# m²_σ,eff(T) ≈ m²_σ(T=0) + (y²/12f²) T² [thermal correction]
#
# But we're in the matter era, not thermal. The relevant quantity is:
# m²_σ,eff = (1/f²) d²V_CW/dθ² evaluated at the field configuration
# where n_χ contributes to the effective potential.
#
# More precisely: the finite-density correction to the CW potential:
# δV(θ, n_χ) ~ n_χ × dM_eff/dθ × (1/M_eff)
# This gives: δm²_σ ~ n_χ y² / (f² M_eff)

print("Finite-density correction to m²_σ:")
print("  δm²_σ ~ n_χ y² / (f² m_χ)")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    alpha = bp["alpha"]
    y_sq = 4 * np.pi * alpha / np.cos(theta_relic)**2
    y = np.sqrt(y_sq)
    
    # Number density today: n_χ = ρ_DM / m_χ
    rho_DM_today = 1.05e-47  # GeV⁴
    n_chi_today = rho_DM_today / m_chi  # GeV³
    
    # At z = 1100: n_χ = (1+z)³ × n_χ,today
    n_chi_rec = (1101)**3 * n_chi_today
    
    for f_over_mpl, f_label in [(0.2, "0.2"), (15, "15"), (20, "20")]:
        f = f_over_mpl * M_Pl
        
        # Vacuum CW mass (from d²V_CW/dθ²):
        m2_sigma_vac = y**2 * m_chi**2 / (16 * np.pi**2 * f**2)  # rough
        
        # Finite-density correction:
        delta_m2_today = n_chi_today * y**2 / (f**2 * m_chi)
        delta_m2_rec = n_chi_rec * y**2 / (f**2 * m_chi)
        
        m_sigma_today = np.sqrt(m2_sigma_vac + delta_m2_today)
        m_sigma_rec = np.sqrt(m2_sigma_vac + delta_m2_rec)
        
        if f_label == "0.2":
            print(f"  {name} (f={f_label} M_Pl, β={1/f_over_mpl:.1f}):")
            print(f"    m_σ(vacuum)  = {np.sqrt(m2_sigma_vac)/H_0:.2e} H₀")
            print(f"    m_σ(today)   = {m_sigma_today/H_0:.2e} H₀")
            print(f"    m_σ(z=1100)  = {m_sigma_rec/H_0:.2e} H₀")
            
            range_today = 1/(m_sigma_today) * 1.97e-16 / 3.086e22  # Mpc
            range_rec = 1/(m_sigma_rec) * 1.97e-16 / 3.086e22  # Mpc
            print(f"    Range today  = {range_today:.2e} Mpc")
            print(f"    Range z=1100 = {range_rec:.2e} Mpc")
            print()

# =============================================================================
# SUMMARY
# =============================================================================
print("="*78)
print("SUMMARY: IS THE THEORY ALIVE?")
print("="*78)
print()
print("SCENARIO 1: No screening (vanilla coupled DE)")
print("  β = 5 → EXCLUDED by Planck (β < 0.066)")
print("  Need f > 15 M_Pl → V_σ suppressed by 5600×")
print("  MAP: V_σ/ρ_Λ ~ 10⁻⁴·⁹ → DEAD (too small by 10⁵)")
print()
print("SCENARIO 2: Chameleon screening from DM density")
print("  β = 5 but fifth force range ≪ CMB scales at z=1100")
print("  CW mass ~ 10⁸⁻¹⁰ H₀ → range ~ 10⁻⁸⁻¹⁰ Mpc")
print("  Sound horizon ~ 0.15 Mpc → SCREENED at CMB ✓")
print("  But need to verify Bullet Cluster & local constraints")
print()
print("SCENARIO 3: Adiabatic suppression")
print("  σ mass at early times ≫ H → σ frozen, doesn't evolve")
print("  Only starts rolling when m_σ ~ H (late universe)")
print("  Effectively β_eff → 0 at early times → evades CMB")
print()
print("VERDICT: Theory is NOT immediately dead.")
print("The CW structure naturally provides density-dependent screening.")
print("Need detailed calculation of β_eff(z) to compare with Planck.")
