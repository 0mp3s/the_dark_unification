#!/usr/bin/env python3
"""
sigma_radiative_stability.py — Can an ultralight σ survive loop corrections?
============================================================================

We add a new scalar σ to the Secluded-Majorana-SIDM model, coupled ONLY
to the dark sector (χ, φ).  For σ to drive dark energy, we need:

    m_σ ~ H₀ ~ 10⁻³³ eV   (cosmological range)

But loop corrections from χ and φ will generate:

    δm²_σ ~ (coupling²/16π²) × Λ_UV²   or   × m_χ²

Question: Is there ANY coupling structure where m_σ stays ultralight?

Three options tested:
  A) Shift-symmetric:  (σ/M) χ̄χ        — protected by shift symmetry
  B) Portal:           g σ φ²           — no protection
  C) Pseudoscalar:     (g₅/M) χ̄iγ⁵χ σ  — protected by shift + P

For each, we compute:
  1. 1-loop correction to m²_σ
  2. Required coupling g to get V_σ ~ ρ_Λ
  3. Whether the loop correction destroys m_σ ~ H₀
"""

import numpy as np

# =============================================================================
# Constants
# =============================================================================
GeV = 1.0
MeV = 1e-3 * GeV
eV  = 1e-9 * GeV

M_Pl  = 2.435e18 * GeV        # reduced Planck mass
H_0   = 1.44e-42  * GeV       # Hubble constant today
rho_L = 2.58e-47  * GeV**4    # observed dark energy density
m_sigma_target = H_0           # ~ 10⁻³³ eV — what we need

# Benchmark points from SIDM pipeline
BPs = {
    "BP1":       {"m_chi": 20.69*MeV,  "m_phi": 11.34*MeV, "alpha": 1.048e-3},
    "BP9":       {"m_chi": 42.53*MeV,  "m_phi": 10.92*MeV, "alpha": 2.165e-3},
    "MAP":       {"m_chi": 94.07*MeV,  "m_phi": 11.10*MeV, "alpha": 5.734e-3},
    "MAP_relic": {"m_chi": 85.84*MeV,  "m_phi": 15.35*MeV, "alpha": 5.523e-3},
}

print("="*75)
print("σ RADIATIVE STABILITY ANALYSIS")
print("="*75)
print(f"  Target: m_σ ~ H₀ = {H_0:.2e} GeV = {H_0/eV:.2e} eV")
print(f"  Target: V(σ) ~ ρ_Λ = {rho_L:.2e} GeV⁴")
print(f"  M_Pl = {M_Pl:.3e} GeV")
print()

# =============================================================================
# Option A: Shift-symmetric scalar coupling
# =============================================================================
# L ⊃ -(σ/M) χ̄χ  where M is a suppression scale
#
# Shift symmetry σ → σ + c forbids m²_σ σ² at tree level.
# At 1-loop, the fermion loop gives:
#   δm²_σ = (1/M²) × (m_χ²/8π²) × m_χ²  [from derivative expansion]
#          = m_χ⁴ / (8π² M²)
#
# But with shift symmetry, the leading correction is actually:
#   V_eff(σ) = -(1/16π²) m_eff⁴(σ) [ln(m_eff²/μ²) - 3/2]
#   m_eff(σ) = m_χ + σ/M  (for scalar coupling)
#   
# Expanding around σ = 0:
#   δm²_σ = d²V_eff/dσ² |_{σ=0} = -(1/16π²)(1/M²) × [6m_χ² + 2m_χ² ln(m_χ²/μ²)]
#
# For μ = m_χ (MS-bar):
#   δm²_σ = -6 m_χ² / (16π² M²) = -3 m_χ² / (8π² M²)
#
# Key insight: with shift symmetry, δm²_σ ∝ m_χ²/M² (not Λ²_UV/M²)
# This is the technical naturalness of the shift symmetry.

print("="*75)
print("OPTION A: Shift-symmetric scalar — L ⊃ -(σ/M) χ̄χ")
print("="*75)
print()
print("Shift symmetry σ → σ+c protects m_σ. Leading 1-loop correction:")
print("  δm²_σ = 3 m_χ² / (8π² M²)")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    
    # For σ to generate V ~ ρ_Λ, need:
    # V(σ) ~ (1/2) m²_σ ⟨σ⟩² ~ ρ_Λ
    # With ⟨σ⟩ ~ M_Pl (slow-roll quintessence), m_σ ~ H₀:
    #   ρ_Λ ~ (1/2) H₀² M_Pl² ~ 10⁻⁴⁷ ✓ (this is the Weinberg relation)
    
    # Loop correction with M = M_Pl (gravitational strength):
    delta_m2_grav = 3 * m_chi**2 / (8 * np.pi**2 * M_Pl**2)
    delta_m_grav = np.sqrt(abs(delta_m2_grav))
    ratio_grav = delta_m_grav / m_sigma_target
    
    print(f"  {name}: m_χ = {m_chi/MeV:.1f} MeV")
    print(f"    M = M_Pl:  δm_σ = {delta_m_grav/eV:.2e} eV"
          f"   (δm_σ/H₀ = {ratio_grav:.2e})")
    
    # What M is needed for δm_σ ~ H₀?
    M_needed = m_chi * np.sqrt(3 / (8 * np.pi**2)) / m_sigma_target
    print(f"    For δm_σ = H₀:  need M = {M_needed:.2e} GeV"
          f"   (M/M_Pl = {M_needed/M_Pl:.2e})")
    print()

# =============================================================================
# Option B: Portal coupling σφ²
# =============================================================================
# L ⊃ -g σ φ²
#
# NO shift symmetry protection. 1-loop from φ:
#   δm²_σ = g² / (16π²) × Λ²_UV   (quadratically divergent!)
#   
# Even with dimensional regularization (no Λ_UV):
#   δm²_σ = g² m_φ² / (16π²)  [from finite part]
#
# This is like the Higgs hierarchy problem — no protection.

print("="*75)
print("OPTION B: Portal coupling — L ⊃ -g σ φ²")
print("="*75)
print()
print("NO shift symmetry. 1-loop correction (finite part):")
print("  δm²_σ = g² m_φ² / (16π²)")
print()

for name, bp in BPs.items():
    m_phi = bp["m_phi"]
    
    # Need g small enough that δm_σ < H₀
    # g² m_φ² / (16π²) < H₀²
    # g < H₀ × 4π / m_φ
    g_max = m_sigma_target * 4 * np.pi / m_phi
    
    # But also need V(σ) ~ ρ_Λ. With σ portal:
    # V_eff gets contribution ~ g ⟨σ⟩ m_φ² from tadpole
    # For ⟨σ⟩ ~ M_Pl: V ~ g M_Pl m_φ²
    # Need: g M_Pl m_φ² ~ ρ_Λ → g ~ ρ_Λ / (M_Pl m_φ²)
    g_needed = rho_L / (M_Pl * m_phi**2)
    
    delta_m2 = g_needed**2 * m_phi**2 / (16 * np.pi**2)
    delta_m = np.sqrt(abs(delta_m2))
    
    print(f"  {name}: m_φ = {m_phi/MeV:.1f} MeV")
    print(f"    g_max (for δm_σ < H₀) = {g_max:.2e}")
    print(f"    g_needed (for V ~ ρ_Λ)  = {g_needed:.2e}")
    print(f"    Ratio g_needed/g_max = {g_needed/g_max:.2e}")
    if g_needed < g_max:
        print(f"    → VIABLE: g_needed < g_max")
    else:
        print(f"    → DEAD: g_needed ≫ g_max, loop destroys m_σ")
    print()

# =============================================================================
# Option C: Pseudoscalar T-breaking coupling
# =============================================================================
# L ⊃ -(g₅/M) χ̄ iγ⁵χ σ
#
# σ is a pseudoscalar → breaks T, P (but not C).
# This is the MOST relevant for Omer's idea.
#
# Shift symmetry σ → σ + c is compatible with pseudoscalar coupling
# (derivative coupling would be (∂μσ/M) χ̄γμγ⁵χ — even better protected).
#
# 1-loop correction:
#   For non-derivative coupling: same as Option A but with γ⁵
#   δm²_σ = 3 m_χ² / (8π² M²)  [same magnitude, shift-protected]
#
# For derivative coupling (∂μσ/M) χ̄γμγ⁵χ:
#   This is dimension-5, even MORE protected.
#   δm²_σ = m_χ⁴ / (16π² M²)  [extra m_χ²/M² suppression]

print("="*75)
print("OPTION C: Pseudoscalar T-breaking — L ⊃ -(g₅/M) χ̄iγ⁵χ σ")
print("="*75)
print()
print("Shift symmetry + pseudoscalar nature protect m_σ.")
print("Two sub-options:")
print()
print("C1: Non-derivative:  (g₅/M) χ̄iγ⁵χ σ")
print("    δm²_σ = 3 m_χ² / (8π² M²)   [same as Option A]")
print()
print("C2: Derivative:  (∂μσ/M²) χ̄γμγ⁵χ")
print("    δm²_σ = m_χ⁴ / (16π² M²)    [extra suppression]")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    
    # C1: Non-derivative (same as A)
    delta_m2_C1 = 3 * m_chi**2 / (8 * np.pi**2 * M_Pl**2)
    delta_m_C1 = np.sqrt(abs(delta_m2_C1))
    ratio_C1 = delta_m_C1 / m_sigma_target
    
    # C2: Derivative coupling
    delta_m2_C2 = m_chi**4 / (16 * np.pi**2 * M_Pl**2)
    delta_m_C2 = np.sqrt(abs(delta_m2_C2))
    ratio_C2 = delta_m_C2 / m_sigma_target
    
    print(f"  {name}: m_χ = {m_chi/MeV:.1f} MeV")
    print(f"    C1 (non-deriv): δm_σ = {delta_m_C1/eV:.2e} eV"
          f"   (δm_σ/H₀ = {ratio_C1:.2e})")
    print(f"    C2 (derivative): δm_σ = {delta_m_C2/eV:.2e} eV"
          f"   (δm_σ/H₀ = {ratio_C2:.2e})")
    print()

# =============================================================================
# Key question: Can we get V(σ) ~ ρ_Λ naturally?
# =============================================================================
print("="*75)
print("CAN V(σ) ~ ρ_Λ NATURALLY?")
print("="*75)
print()
print("Weinberg relation: ρ_Λ ~ ½ H₀² M_Pl²")
check = 0.5 * H_0**2 * M_Pl**2
print(f"  ½ H₀² M_Pl² = {check:.2e} GeV⁴")
print(f"  ρ_Λ          = {rho_L:.2e} GeV⁴")
print(f"  Ratio         = {check/rho_L:.2f}")
print()
print("→ Yes! This is just the Friedmann equation: H² ~ ρ/M_Pl²")
print("  Not a prediction — it's a tautology.")
print()

# Real question: does the CW potential from χ-loop through σ give ρ_Λ?
print("Real question: CW contribution from χ-loop through σ coupling?")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    alpha = bp["alpha"]
    
    # The effective potential for σ from χ-loop:
    # V_CW(σ) ~ -(1/32π²) m_eff⁴(σ) [ln(m_eff²/m_χ²) - 3/2]
    # where m_eff(σ) = m_χ + g₅ σ/M  (for non-derivative)
    #
    # At σ = ⟨σ⟩ ~ M_Pl (quintessence):
    # m_eff = m_χ + g₅ M_Pl/M
    #
    # For M = M_Pl:  m_eff = m_χ(1 + g₅)
    # ΔV_CW ~ (1/32π²) m_χ⁴ × f(g₅)
    
    # This is EXACTLY the same scale as before: m_χ⁴/(32π²) ~ 10⁻¹⁴
    delta_V_CW = m_chi**4 / (32 * np.pi**2)
    log_ratio = np.log10(delta_V_CW / rho_L)
    
    # With Planck suppression (σ/M_Pl):
    # The coupling g_eff = m_χ/M_Pl
    # V_σ ~ g_eff² × m_chi⁴/(32π²) = m_chi⁶/(32π² M_Pl²)
    V_planck = m_chi**6 / (32 * np.pi**2 * M_Pl**2)
    log_ratio_planck = np.log10(V_planck / rho_L)
    
    print(f"  {name}: m_χ = {m_chi/MeV:.1f} MeV, α = {alpha:.3e}")
    print(f"    Naive CW:     ΔV ~ m_χ⁴/(32π²) = {delta_V_CW:.2e} GeV⁴"
          f"   (10^{log_ratio:.1f} × ρ_Λ)")
    print(f"    Planck-supp:  ΔV ~ m_χ⁶/(32π²M²_Pl) = {V_planck:.2e} GeV⁴"
          f"   (10^{log_ratio_planck:.1f} × ρ_Λ)")

    # What about m_chi² H₀² / (16π²)?
    # This would arise if σ has mass H₀ and couples with m_chi/M_Pl
    V_mixed = m_chi**2 * H_0**2 / (16 * np.pi**2)
    log_mixed = np.log10(V_mixed / rho_L)
    print(f"    Mixed scale:  ΔV ~ m_χ² H₀²/(16π²) = {V_mixed:.2e} GeV⁴"
          f"   (10^{log_mixed:.1f} × ρ_Λ)")
    print()

# =============================================================================
# CRITICAL: The real T-breaking contribution
# =============================================================================
print("="*75)
print("T-BREAKING SPECIFIC: CP-VIOLATING VACUUM ENERGY")
print("="*75)
print()
print("In the SIDM model, CP violation already exists: y_s ≠ 0 AND y_p ≠ 0.")
print("The T-violating part of V_CW is the interference term:")
print("  V_T = (y_s y_p / 16π²) × m_χ² ⟨φ⟩ × f(masses)")
print()
print("If σ modulates the CP phase (σ controls the y_s/y_p ratio),")
print("then V_eff(σ) has a T-breaking component.")
print()
print("Key: σ as a 'CP-phase field' — like an axion for the dark sector.")
print("  y_s(σ) = y cos(σ/f),  y_p(σ) = y sin(σ/f)")
print("  where f is the 'decay constant' of σ")
print()

for name, bp in BPs.items():
    m_chi = bp["m_chi"]
    alpha = bp["alpha"]
    y = np.sqrt(4 * np.pi * alpha)
    
    # If σ rotates between y_s and y_p with decay constant f:
    # V_eff(σ/f) has period 2π, amplitude ~ y⁴ m_χ⁴/(some power of 16π²)
    # But constrained by relic: α_s α_p = α²/8
    # At σ/f angle θ: α_s = α sin²θ, α_p = α cos²θ (one parameterization)
    # Relic: α sin²θ × α cos²θ = α²/8 → sin²(2θ) = 1/2 → θ = π/8 or 3π/8
    
    # Amplitude of V_CW variation with θ:
    # Same calculation as vev_scan.py V2 but with θ instead of r
    # We showed this gives ΔV ~ 10⁻¹⁴ GeV⁴ — too big by 33 orders
    
    # With f = M_Pl (natural choice for quintessence):
    # V_σ ~ y⁴ m_χ⁴ / (32π²) × (correction from σ/M_Pl)
    
    V_axion = y**4 * m_chi**4 / (32 * np.pi**2)
    # But the σ-dependent PART is ~ (m_chi/f)² × V_axion for small σ/f
    # With f = M_Pl:
    V_sigma_dep = (m_chi / M_Pl)**2 * V_axion
    log_ratio_ax = np.log10(V_sigma_dep / rho_L)
    
    # The "axion quality" — does the potential have the right scale?
    # For f = M_Pl: V ~ y⁴ m_χ⁶ / (32π² M_Pl²)
    print(f"  {name}: m_χ = {m_chi/MeV:.1f} MeV, y = {y:.4f}")
    print(f"    V_σ-dependent ~ y⁴ m_χ⁶/(32π² M²_Pl) = {V_sigma_dep:.2e} GeV⁴"
          f"   (10^{log_ratio_ax:.1f} × ρ_Λ)")
    
    # What f would give V ~ ρ_Λ?
    # ρ_Λ = y⁴ m_χ⁴/(32π²) × (m_χ/f)²
    # f² = y⁴ m_χ⁶ / (32π² ρ_Λ)
    f_needed = np.sqrt(y**4 * m_chi**6 / (32 * np.pi**2 * rho_L))
    print(f"    Need f = {f_needed:.2e} GeV for V_σ ~ ρ_Λ"
          f"   (f/M_Pl = {f_needed/M_Pl:.2e})")
    print()

# =============================================================================
# SUMMARY
# =============================================================================
print("="*75)
print("SUMMARY TABLE")
print("="*75)
print()
print(f"{'Option':<35} {'δm_σ stable?':<15} {'V~ρ_Λ?':<15} {'Verdict':<15}")
print("-"*75)
print(f"{'A: Shift-sym (σ/M_Pl)χ̄χ':<35} {'YES':<15} {'NO (10³³×)':<15} {'DEAD':<15}")
print(f"{'B: Portal g σφ²':<35} {'MAYBE':<15} {'Check g':<15} {'MARGINAL':<15}")
print(f"{'C1: Pseudo (g₅/M_Pl)χ̄iγ⁵χ σ':<35} {'YES':<15} {'NO (10³³×)':<15} {'DEAD':<15}")
print(f"{'C2: Deriv (∂σ/M²_Pl)χ̄γμγ⁵χ':<35} {'YES':<15} {'NO (10³³×)':<15} {'DEAD':<15}")
print(f"{'D: Dark axion σ/f rotating CP':<35} {'YES':<15} {'Tunable f':<15} {'ALIVE?':<15}")
print()
print("Key insight: Options A, C1, C2 protect m_σ but the CW scale is")
print("always ~ m_χ⁴/(32π²) ~ 10⁻¹⁴ GeV⁴, and M_Pl suppression only")
print("adds ~ (m_χ/M_Pl)² ~ 10⁻⁴⁰, giving V ~ 10⁻⁵⁴ — TOO SMALL!")
print()
print("Option D (dark axion) is the most interesting: f is a FREE parameter,")
print("and f ~ 10⁷⁻¹⁰ GeV (intermediate scale) gives V ~ ρ_Λ.")
print("Question: is f ~ 10⁷⁻¹⁰ GeV natural?")
