#!/usr/bin/env python3
"""
theta_topological.py — Is θ_dark = arctan(1/√8) a topological parameter?
=========================================================================

Instead of treating σ as a dynamical field that rolls in V_CW,
explore the possibility that θ = σ/f is a FIXED topological parameter,
analogous to θ_QCD in QCD.

Key questions:
  1. Does tan²θ = 1/8 arise from a discrete symmetry?
  2. What is the algebraic/group-theoretic meaning of 1/8?
  3. Can the Witten effect connect θ to magnetic charges?
  4. What UV structure fixes θ at this value?
  5. Is 19.47° a "special" angle in any known mathematical context?
"""

import numpy as np
from fractions import Fraction
import warnings
warnings.filterwarnings('ignore')

MeV = 1e-3; GeV = 1.0; eV = 1e-9
M_Pl = 2.435e18; rho_L = 2.58e-47

theta_relic = np.arctan(1/np.sqrt(8))

print("="*78)
print("θ_dark AS TOPOLOGICAL PARAMETER — EXPLORATION")
print("="*78)

# ============================================================================
# PART 1: The number 1/8 — algebraic meaning
# ============================================================================
print(f"\n{'='*78}")
print("PART 1: THE NUMBER 1/8 — WHERE DOES IT COME FROM?")
print("="*78)

print(f"""
  The constraint chain:
    SIDM:  σ_T/m = πα²/m³_χ  →  α_s = α
    Relic:  ⟨σv⟩ = 2πα_sα_p/m²_χ  →  α_s α_p = α²/8

  From these two:
    α_p = α/8

  The factor 1/8 comes from the relic abundance calculation:
    ⟨σv⟩_relic ≈ 2×10⁻²⁶ cm³/s ∝ α²/m²_χ
    But the specific factor depends on:
""")

# Where does 1/8 come from in the cross section?
# ⟨σv⟩ for χχ → φφ with scalar (y_s) and pseudoscalar (y_p) couplings
# Majorana fermion: additional factor of 1/2 from identical particles
# s-wave part: proportional to y_s² y_p² / m²_χ
# The exact coefficient depends on the matrix element

print("  For Majorana χ → φφ (s-wave):")
print("    |M|² ∝ y_s² y_p²  (interference of scalar × pseudoscalar)")
print("    The factor 1/4 from Majorana statistics")
print("    The phase space factor gives another 1/(8π)")
print("    Total: ⟨σv⟩ = (y_s y_p)² / (16π m²_χ)")
print()
print("  With y_s = y cosθ, y_p = y sinθ:")
print("    ⟨σv⟩ = y⁴ sin²(2θ) / (256π m²_χ)")
print()
print("  But also: ⟨σv⟩ = 2π α_s α_p / m²_χ  (in terms of fine structure)")
print("    where α_i = y²_i/(4π)")
print()
print("  So: 2π × (y²cos²θ/4π) × (y²sin²θ/4π) / m²_χ = y⁴ sin²(2θ)/(32π m²_χ)")
print()

# The 1/8 comes from requiring this equals the SIDM cross section divided by 8
# Let's trace it exactly
print("  The constraint is:")
print("    α_s α_p = ⟨σv⟩_needed × m²_χ / (2π)")
print("    α² = σ_T/m × m³_χ / π  (from SIDM)")
print()
print("  The ratio α_p/α = 1/8 comes from:")
print("    ⟨σv⟩_needed / (α × 2π/m²_χ) = 1/8")
print()

# ============================================================================
# PART 2: Is 19.47° special?
# ============================================================================
print(f"\n{'='*78}")
print("PART 2: IS 19.47° A SPECIAL ANGLE?")
print("="*78)

theta_deg = np.degrees(theta_relic)
print(f"\n  θ = arctan(1/√8) = {theta_deg:.4f}°")
print(f"  cos²θ = 8/9 = {np.cos(theta_relic)**2:.6f}")
print(f"  sin²θ = 1/9 = {np.sin(theta_relic)**2:.6f}")
print(f"  tan²θ = 1/8 = {np.tan(theta_relic)**2:.6f}")

print(f"\n  ⚡ KEY: cos²θ = 8/9, sin²θ = 1/9")
print(f"  This means the coupling splits as 8:1 between scalar and pseudoscalar!")

# Check known special angles
print(f"\n  Comparison with known special angles:")
special_angles = {
    "Tetrahedral (magic) angle": np.degrees(np.arctan(np.sqrt(2))),
    "Cabibbo angle": 13.02,
    "Weinberg angle (sin²θ_W=0.231)": np.degrees(np.arcsin(np.sqrt(0.231))),
    "arctan(1/√8) = our θ": theta_deg,
    "arccos(1/3) ≈ tetrahedral": np.degrees(np.arccos(1/3)),
    "arctan(1/3)": np.degrees(np.arctan(1/3)),
}
for name, angle in sorted(special_angles.items(), key=lambda x: x[1]):
    print(f"    {name}: {angle:.2f}°")

# tetrahedral angle connection
print(f"\n  === TETRAHEDRAL CONNECTION ===")
theta_tet = np.arccos(1/3)  # 70.53° — angle between vertices of tetrahedron
print(f"  Tetrahedral angle: arccos(1/3) = {np.degrees(theta_tet):.2f}°")
print(f"  Our angle:         arctan(1/√8) = {theta_deg:.2f}°")
print(f"  Sum:               {np.degrees(theta_tet) + theta_deg:.2f}° = 90°? "
      f"{'YES!' if abs(np.degrees(theta_tet) + theta_deg - 90) < 0.01 else 'No'}")

# Check: is arctan(1/√8) = π/2 - arccos(1/3)?
# arctan(1/√8): tanθ = 1/√8 → sinθ = 1/3, cosθ = √8/3 = 2√2/3
# arccos(1/3): cosφ = 1/3 → sinφ = √(1-1/9) = √(8/9) = 2√2/3
# So sinθ = cosφ → θ + φ = π/2 ✓
print(f"\n  sinθ = 1/3 = {np.sin(theta_relic):.6f}")
print(f"  cosθ = 2√2/3 = {2*np.sqrt(2)/3:.6f} = {np.cos(theta_relic):.6f}")
print(f"\n  ⚡ θ_dark = arcsin(1/3) exactly!")
print(f"  ⚡ θ_dark + θ_tetrahedral = 90° exactly!")
print(f"  ⚡ The dark angle is the COMPLEMENT of the tetrahedral angle!")

# ============================================================================
# PART 3: Group-theoretic meaning of 1/3
# ============================================================================
print(f"\n{'='*78}")
print("PART 3: WHY sinθ = 1/3? — GROUP THEORY")
print("="*78)

print(f"""
  sinθ = 1/3 is deeply connected to GEOMETRY:

  In a regular tetrahedron inscribed in a sphere of radius R:
    - Height h = 4R/3
    - Distance from center to face = R/3
    - cos(vertex angle) = 1/3
    - sin(face-center angle) = 1/3  ← THIS IS OUR θ!

  The tetrahedron is the simplest 3D Platonic solid.
  Its symmetry group is S₄ (permutation of 4 vertices) = 24 elements.
  The alternating subgroup A₄ has 12 elements.
""")

# S4 and A4 are well-known discrete symmetry groups used in neutrino physics!
print("  ⚡ A₄ is THE discrete symmetry used in neutrino mixing!")
print("  (tribimaximal mixing: sin²θ₁₂ = 1/3, exactly the same number!)")
print()
print("  In tribimaximal mixing (Harrison-Perkins-Scott 2002):")
print(f"    sin²θ₁₂ = 1/3  →  θ₁₂ = {np.degrees(np.arcsin(np.sqrt(1/3))):.2f}°")
print(f"    Our sin²θ_dark = 1/9 = (1/3)²")
print(f"    Or: sinθ_dark = 1/3 = √(sin²θ₁₂)")
print()
print("  This suggests a COMMON discrete symmetry origin!")

# ============================================================================
# PART 4: Z_N symmetry that fixes θ
# ============================================================================
print(f"\n{'='*78}")
print("PART 4: WHICH DISCRETE SYMMETRY FIXES θ?")
print("="*78)

print(f"""
  If θ is a topological angle (like θ_QCD), it can be fixed by:

  Option A: Z_N orbifold
    σ → σ + 2πf/N  is a gauge symmetry
    The vacuum is at σ/f = 2πk/N for integer k
    Need: arctan(1/√8) = 2π k/N for some (k,N)

  Testing:""")

# Check if θ_relic / (2π) is a rational number
ratio = theta_relic / (2 * np.pi)
print(f"    θ_relic/(2π) = {ratio:.8f}")
print(f"    Closest fractions:")
for N in range(2, 100):
    for k in range(1, N):
        if abs(k/N - ratio) < 0.001:
            print(f"      k={k}, N={N}: {k}/{N} = {k/N:.8f}, "
                  f"error = {abs(k/N - ratio)*360:.2f}°")

print(f"""
  → θ_relic/(2π) is NOT a simple rational fraction.
  → Z_N orbifold does NOT naturally give θ = arctan(1/√8).
  
  Option B: Anomaly matching
    In QCD: θ is fixed to 0 by the axion mechanism (Peccei-Quinn).
    Here:   θ is fixed to arctan(1/√8) by... what?

    Key difference from QCD: we don't want θ=0!
    We want a NON-ZERO θ angle.
    This is like asking "why is the QCD θ angle non-zero?"

  Option C: Discrete Z₂ × Z₂ with specific charges
    If there are TWO Z₂ symmetries that constrain y_s and y_p independently,
    the ratio y²_p/y²_s could be fixed at 1/8.
""")

# ============================================================================
# PART 5: The Witten effect — θ angle → magnetic charge
# ============================================================================
print(f"{'='*78}")
print("PART 5: WITTEN EFFECT IN THE DARK SECTOR")
print("="*78)

print(f"""
  The Witten effect (1979): In a gauge theory with θ-angle,
  a magnetic monopole of charge g acquires an ELECTRIC charge:
  
    q_induced = -eθ/(2π)
  
  A pure magnetic monopole (g) becomes a dyon (e·θ/2π, g).
  
  In our EM duality language:
    - "Electric" = scalar coupling α_s
    - "Magnetic" = pseudoscalar coupling α_p
    - θ_dark mixes them
  
  If we interpret θ_dark as a Witten-type angle:
    α_p/α_s = tan²θ = 1/8
    "Magnetic charge"² / "Electric charge"² = 1/8
    
  Dirac quantization: e·g = 2πn → α_s·α_p = n²/4 (in natural units)
  In our case: α_s·α_p = α²/8
  
  This is NOT standard Dirac quantization (which gives integers).
  But it COULD come from a non-abelian generalization.
""")

# Check: is α²/8 consistent with any quantization?
alpha_val = 5.734e-3  # MAP
product = alpha_val**2 / 8
print(f"  For MAP: α²/8 = {product:.4e}")
print(f"  Dirac: n²/4  → n² = {product*4:.4e}  → n = {np.sqrt(product*4):.4e}")
print(f"  → Not integer. But α itself is a running coupling, not quantized.")
print()

# The quantization is in the TOPOLOGICAL sector
# The GROUP theory fixes the RATIO, not the absolute value
print("  KEY INSIGHT: The Witten effect fixes the RATIO α_p/α_s,")
print("  not the absolute values. The absolute scale is set by SIDM phenomenology.")
print(f"  The ratio = 1/8 = 1/2³")
print()
print("  Why 2³?")
print("  - Majorana fermion has 2 real dof (vs 4 for Dirac)")
print("  - s-wave has ℓ=0, total spin S can be 0 or 1")
print("  - For Majorana: only S=0 (antisymmetric) → factor 1/2")
print("  - Phase space in 3D → factor 1/2 per dimension? → 1/2³ = 1/8")

# ============================================================================
# PART 6: A₄ symmetry construction
# ============================================================================
print(f"\n{'='*78}")
print("PART 6: A₄ CONSTRUCTION — CAN IT FIX θ?")
print("="*78)

print(f"""
  A₄ is the symmetry group of the tetrahedron (12 elements).
  It has irreps: 1, 1', 1'', 3
  
  The 3-dimensional representation decomposes as:
    3 → real components (x, y, z) on tetrahedron vertices
  
  If χ transforms as 1 under A₄, and σ transforms as 1' or 1'':
    - y_s and y_p are related by A₄ Clebsch-Gordan coefficients
    - The VEV alignment fixes the ratio
  
  Classic A₄ alignment:
    ⟨φ₃⟩ = v(1, 1, 1)/√3  →  gives sin²θ₁₂ = 1/3 in neutrino mixing
  
  For our dark sector:
    If the dark Higgs ⟨Φ⟩ transforms as 3 under A₄:
    ⟨Φ⟩ = v(1, 0, 0)  →  breaks A₄ → Z₃
    ⟨Φ⟩ = v(1, 1, 1)/√3  →  breaks A₄ → Z₃  (different embedding)
  
  The ratio sin²θ = 1/9 = (1/3)² could come from:
    - Two sequential A₄ breakings
    - A₄ × Z₃ with specific charge assignments
    - S₄ (parent group of A₄) with its own CG coefficients
""")

# What coupling structure gives sin²θ = 1/9?
# If y_s and y_p arise from A4 CG coefficients:
# y_s ∝ C_s × v, y_p ∝ C_p × v
# sin²θ = y²_p/(y²_s + y²_p) = C²_p/(C²_s + C²_p) = 1/9
# → C²_s/C²_p = 8 → C_s/C_p = 2√2

print("  Required: C_s/C_p = 2√2 (ratio of Clebsch-Gordan coefficients)")
print(f"  In A₄: the 3⊗3 decomposition gives:")
print(f"    3 ⊗ 3 = 1 ⊕ 1' ⊕ 1'' ⊕ 3_s ⊕ 3_a")
print(f"  The CG coefficients for (1,1,1) alignment:")
print(f"    Scalar singlet (1): coefficient = 1/√3 per component")
print(f"    This doesn't directly give 2√2.")
print()
print("  BUT in S₄ (24 elements, the FULL tetrahedral group):")
print("    S₄ has irreps: 1, 1', 2, 3, 3'")
print("    The 2-dimensional irrep is special!")
print("    In the 2⊗3 product, new CG coefficients appear.")
print()

# ============================================================================
# PART 7: The simplest UV completion
# ============================================================================
print(f"{'='*78}")
print("PART 7: SIMPLEST UV COMPLETION — FIXED θ WITHOUT DYNAMICS")
print("="*78)

print(f"""
  Instead of σ being a field, consider:
  
  MODEL: Two heavy dark fermions Ψ₁, Ψ₂ at scale Λ_UV
  
    L_UV = -Ψ̄₁(g₁ + ig₁'γ⁵)Ψ₁ φ - Ψ̄₂(g₂ + ig₂'γ⁵)Ψ₂ φ + ...
  
  At low energy (below Λ_UV), integrating out Ψ₁, Ψ₂ generates
  effective couplings for χ:
    y_s = g₁ cos α₁ + g₂ cos α₂  (sum of scalar parts)
    y_p = g₁ sin α₁ + g₂ sin α₂  (sum of pseudo parts)
  
  The ratio y_p/y_s is then FIXED by UV couplings, not by a rolling field.
  
  θ_dark = arctan(y_p/y_s) is a DERIVED PARAMETER.
  It doesn't roll. It doesn't have a potential. It IS what it IS.
  
  The CW potential creates V(σ), but there IS no σ — just fixed couplings.
  V_CW problem: SOLVED (by not existing).
""")

# Can this approach give the right DE?
print("  But where does dark energy come from in this picture?")
print()
print("  The vacuum energy from the dark sector is:")
print("  V_vac = V_CW(θ_fixed) — a CONSTANT, not dynamical")
print("  This is just a contribution to the cosmological constant!")
print()

# Compute it
for name, m_chi_val in [("BP1", 20.69e-3), ("MAP", 94.07e-3)]:
    y_sq = 4 * np.pi * 5.734e-3 / np.cos(theta_relic)**2  # approximate
    y_val = np.sqrt(y_sq)
    v_phi = 0.5 * 11.10e-3
    M2 = m_chi_val**2 + m_chi_val * y_val * np.cos(theta_relic) * v_phi + y_val**2 * v_phi**2 / 4
    V_cw = -(1/(32*np.pi**2)) * M2**2 * (np.log(M2/m_chi_val**2) - 1.5)
    print(f"  {name}: V_CW(θ_relic) = {V_cw:.3e} GeV⁴  (ρ_Λ = {rho_L:.3e})")
    print(f"         V_CW / ρ_Λ = 10^{np.log10(abs(V_cw)/rho_L):.1f}")

print(f"""
  V_CW ~ 10⁻⁷ GeV⁴ while ρ_Λ ~ 10⁻⁴⁷ GeV⁴.
  Gap of 10⁴⁰ — this is the CC problem again.
  
  BUT: this V_CW is the TOTAL potential, not the θ-dependent part.
  The θ-dependent variation is much smaller:
""")

# The θ-dependent part
m_chi_val = 94.07e-3
y_sq = 4 * np.pi * 5.734e-3 / np.cos(theta_relic)**2
y_val = np.sqrt(y_sq)
v_phi = 0.5 * 11.10e-3

V_at_0 = -(1/(32*np.pi**2)) * (m_chi_val**2 + m_chi_val*y_val*v_phi + y_val**2*v_phi**2/4)**2 * \
         (np.log((m_chi_val**2 + m_chi_val*y_val*v_phi + y_val**2*v_phi**2/4)/m_chi_val**2) - 1.5)
V_at_relic = -(1/(32*np.pi**2)) * (m_chi_val**2 + m_chi_val*y_val*np.cos(theta_relic)*v_phi + y_val**2*v_phi**2/4)**2 * \
             (np.log((m_chi_val**2 + m_chi_val*y_val*np.cos(theta_relic)*v_phi + y_val**2*v_phi**2/4)/m_chi_val**2) - 1.5)

delta_V = abs(V_at_0 - V_at_relic)
print(f"  MAP: ΔV_CW(0° → 19.47°) = {delta_V:.3e} GeV⁴")
print(f"       ΔV / ρ_Λ = 10^{np.log10(delta_V/rho_L):.1f}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*78}")
print("SUMMARY: TOPOLOGICAL θ_dark EXPLORATION")
print("="*78)

print(f"""
  FINDINGS:

  1. GEOMETRY: θ_dark = arcsin(1/3) — complement of tetrahedral angle!
     sin²θ = 1/9, cos²θ = 8/9 — the coupling splits 8:1.
     
  2. NEUTRINO CONNECTION: sin²θ₁₂(tribimaximal) = 1/3 = 3 × sin²θ_dark
     Same A₄ symmetry could be responsible for both.
     
  3. Z_N: θ_relic/(2π) is NOT a simple rational → Z_N orbifold won't work.
  
  4. A₄/S₄: The tetrahedral groups naturally produce 1/3 and related fractions.
     Could fix θ_dark through CG coefficients and VEV alignment.
     
  5. WITTEN EFFECT: θ_dark as gauge θ-angle gives the right structure
     but α²/8 is not Dirac-quantized.
  
  6. UV COMPLETION: If θ is not a field but a derived ratio of UV couplings,
     V_CW is irrelevant (no field to roll). The angle is fixed.
     
  7. CC PROBLEM: Even with fixed θ, the vacuum energy from V_CW is
     10⁴⁰ × ρ_Λ. The θ-dependent part is smaller but still huge (10³⁸ × ρ_Λ).

  STATUS:
    ✅ θ_dark has deep geometric meaning (tetrahedral angle complement)
    ✅ A₄/S₄ symmetry could provide the mechanism
    ✅ Fixed-θ approach eliminates V_CW rolling problem
    ⚠️ CC problem remains (V_CW too large as cosmological constant)
    ❌ Dark energy still needs an explanation
    
  NEXT STEPS:
    → Construct explicit A₄ dark sector model
    → Check if A₄ breaking pattern gives exactly sin²θ = 1/9
    → Explore connection to neutrino sector (A₄ is already used there!)
    → Address CC: maybe V_CW is cancelled by tree-level counter-term
""")
