#!/usr/bin/env python3
"""
background_cosmology.py  --  Friedmann + Klein-Gordon with V_A4(theta)
=====================================================================

Solves the coupled ODE system:

    d theta / dN = pi_theta
    d pi_theta / dN = -(3 - epsilon) pi_theta  -  (1/f^2) V'(theta) / H^2

    H^2 = (rho_r + rho_m + rho_sigma) / (3 M_Pl^2)
    rho_sigma = (1/2) f^2 H^2 pi_theta^2  +  V(theta)

where N = ln(a), theta = sigma/f, pi_theta = d theta / dN.

Potential:  V_A4(theta) = -A cos(theta) + B cos(3 theta)
with A/B = 39/5 (from A4 group theory) and B = Lambda_d^4.

Inputs:  { Lambda_d, theta_i, f }  +  Omega_chi h^2 = 0.120  (from Paper 1)
Outputs: H_0, H(z), w(z), Omega_DE(z), d_L(z)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Physical constants (from centralized config)
# =============================================================================
from config import (M_Pl as _M_Pl_cfg, H_100_GEV as _H100_cfg,
                    OMEGA_B_H2 as _OB_cfg, N_EFF as _NEFF_cfg,
                    T_CMB_eV, eV as _eV_cfg, H0_PLANCK_KMS as _H0KMS_cfg,
                    theta_relic as _theta_cfg, AB_RATIO)

M_PL = _M_Pl_cfg
T_CMB_GEV = T_CMB_eV * _eV_cfg
H_100_GEV = _H100_cfg
OMEGA_B_H2 = _OB_cfg
N_EFF = _NEFF_cfg
eV = _eV_cfg

# Radiation density
_RHO_GAMMA = np.pi**2 / 15.0 * T_CMB_GEV**4
_NU_FACTOR = 1.0 + N_EFF * 7.0/8.0 * (4.0/11.0)**(4.0/3.0)
_RHO_UNIT = 3.0 * M_PL**2 * H_100_GEV**2        # rho_crit / h^2
OMEGA_R_H2 = _RHO_GAMMA * _NU_FACTOR / _RHO_UNIT  # ~4.15e-5

# Reference values
H0_PLANCK_KMS = _H0KMS_cfg
H0_PLANCK_GEV = H0_PLANCK_KMS / 100.0 * H_100_GEV

# Theta relic from A4 CG coefficients: g_p/g_s = 1/3
THETA_RELIC = _theta_cfg  # = 18.4349 deg, sin^2 = 1/10

# =============================================================================
# A4 Potential: V(theta) = -A cos(theta) + B cos(3 theta)
# with A/B = 39/5, B = Lambda_d^4
# =============================================================================
def V_A4(theta, Lambda_d4):
    """V_A4(theta) with A = (39/5) Lambda_d^4, B = Lambda_d^4."""
    return -(39.0/5.0) * Lambda_d4 * np.cos(theta) + Lambda_d4 * np.cos(3.0 * theta)

def dV_A4_dtheta(theta, Lambda_d4):
    """dV/d theta."""
    return (39.0/5.0) * Lambda_d4 * np.sin(theta) - 3.0 * Lambda_d4 * np.sin(3.0 * theta)

def d2V_A4_dtheta2(theta, Lambda_d4):
    """d^2 V / d theta^2."""
    return (39.0/5.0) * Lambda_d4 * np.cos(theta) - 9.0 * Lambda_d4 * np.cos(3.0 * theta)

# Normalize: shift V so that V(theta_relic) = 0
# The DE energy is V(theta) - V(theta_relic) > 0 for theta != theta_relic
V_AT_MIN = V_A4(THETA_RELIC, 1.0)  # per Lambda_d4

def V_DE(theta, Lambda_d4):
    """Potential energy relative to the minimum at theta_relic."""
    return V_A4(theta, Lambda_d4) - V_AT_MIN * Lambda_d4

def dV_DE_dtheta(theta, Lambda_d4):
    """dV_DE/d theta (same as dV_A4/d theta since constant drops)."""
    return dV_A4_dtheta(theta, Lambda_d4)


# =============================================================================
# ODE system in e-fold time N = ln(a)
# =============================================================================
def ode_rhs(N, state, f, Lambda_d4, rho_r0, rho_m0):
    """
    state = [theta, pi_theta]
    pi_theta = d theta / dN

    sigma = f * theta,  so d sigma / dN = f * pi_theta
    rho_sigma = (1/2) f^2 H^2 pi_theta^2  +  V(theta)

    Friedmann:  H^2 (3 M_Pl^2 - f^2 pi^2 / 2) = rho_r + rho_m + V
    epsilon = -dH/dN / H = (4/3 rho_r + rho_m + f^2 H^2 pi^2) / (2 M_Pl^2 H^2)
    Klein-Gordon: pi' = -(3-eps) pi - V'(theta) / (f^2 H^2)
    """
    theta, pi_th = state
    a = np.exp(N)

    rho_r = rho_r0 * a**(-4)
    rho_m = rho_m0 * a**(-3)
    V = V_DE(theta, Lambda_d4)
    dV_dth = dV_DE_dtheta(theta, Lambda_d4)

    denom = 3.0 * M_PL**2 - 0.5 * f**2 * pi_th**2
    if denom <= 0:
        return [0.0, 0.0]

    H2 = (rho_r + rho_m + V) / denom
    if H2 <= 0:
        return [0.0, 0.0]

    eps = (4.0/3.0 * rho_r + rho_m + f**2 * H2 * pi_th**2) / (2.0 * M_PL**2 * H2)

    dtheta_dN = pi_th
    dpi_dN = -(3.0 - eps) * pi_th - dV_dth / (f**2 * H2)

    return [dtheta_dN, dpi_dN]


# =============================================================================
# Solver
# =============================================================================
def solve_background(Lambda_d_eV, theta_i, f_over_MPl, omega_chi_h2=0.120,
                     T_RH_GeV=1e5, N_points=5000):
    """
    Solve background cosmology.

    Parameters
    ----------
    Lambda_d_eV  : dark QCD scale in meV  (e.g. 2.0)
    theta_i      : initial misalignment angle [rad]
    f_over_MPl   : f / M_Pl  (e.g. 1.8)
    omega_chi_h2 : DM relic density (default 0.120)
    T_RH_GeV     : reheating temperature [GeV]
    N_points     : number of output points

    Returns
    -------
    dict with all results
    """
    Lambda_d = Lambda_d_eV * 1e-3 * eV   # convert meV -> GeV
    Lambda_d4 = Lambda_d**4
    f = f_over_MPl * M_PL

    # sigma mass at the minimum
    m_sigma = np.sqrt(d2V_A4_dtheta2(THETA_RELIC, Lambda_d4)) / f
    # For reference: GMOR-like mass
    m_gmor = Lambda_d**2 / f

    # Today's densities (at a=1)
    rho_r0 = OMEGA_R_H2 * _RHO_UNIT
    rho_m0 = (omega_chi_h2 + OMEGA_B_H2) * _RHO_UNIT

    # Initial conditions
    g_star_S_RH = 106.75
    g_star_S_0 = 3.91
    a_RH = (T_CMB_GEV / T_RH_GeV) * (g_star_S_0 / g_star_S_RH)**(1.0/3.0)
    N_RH = np.log(a_RH)

    # Integrate from N_RH to N=0 (today)
    N_span = [N_RH, 0.0]
    N_eval = np.linspace(N_RH, 0.0, N_points)

    sol = solve_ivp(
        ode_rhs, N_span,
        [theta_i, 0.0],   # theta_i, pi_theta=0 (frozen)
        args=(f, Lambda_d4, rho_r0, rho_m0),
        method='RK45', rtol=1e-12, atol=1e-15,
        t_eval=N_eval, dense_output=True, max_step=0.5,
    )

    if not sol.success:
        return {'success': False, 'message': sol.message}

    # Extract arrays
    N_arr = sol.t
    a_arr = np.exp(N_arr)
    z_arr = 1.0 / a_arr - 1.0
    theta_arr = sol.y[0]
    pi_arr = sol.y[1]

    # Compute H(N), w(N), Omega_DE(N)
    H_arr = np.zeros_like(N_arr)
    w_arr = np.zeros_like(N_arr)
    Omega_DE_arr = np.zeros_like(N_arr)
    Omega_m_arr = np.zeros_like(N_arr)
    Omega_r_arr = np.zeros_like(N_arr)

    for i in range(len(N_arr)):
        a = a_arr[i]
        th = theta_arr[i]
        pi_th = pi_arr[i]

        rho_r = rho_r0 * a**(-4)
        rho_m = rho_m0 * a**(-3)
        V = V_DE(th, Lambda_d4)

        denom = 3.0 * M_PL**2 - 0.5 * f**2 * pi_th**2
        if denom <= 0:
            H_arr[i] = 0
            continue

        H2 = (rho_r + rho_m + V) / denom
        if H2 <= 0:
            H_arr[i] = 0
            continue

        H_arr[i] = np.sqrt(H2)

        # sigma energy and pressure
        KE = 0.5 * f**2 * H2 * pi_th**2
        rho_sigma = KE + V
        P_sigma = KE - V
        w_arr[i] = P_sigma / rho_sigma if rho_sigma > 1e-100 else -1.0

        rho_tot = rho_r + rho_m + rho_sigma
        Omega_DE_arr[i] = rho_sigma / rho_tot if rho_tot > 0 else 0
        Omega_m_arr[i] = rho_m / rho_tot if rho_tot > 0 else 0
        Omega_r_arr[i] = rho_r / rho_tot if rho_tot > 0 else 0

    # Today's values (N=0, last element)
    H0_GeV = H_arr[-1]
    H0_kms = H0_GeV / H_100_GEV * 100.0
    h = H0_kms / 100.0
    theta_0 = theta_arr[-1]
    w_0 = w_arr[-1]
    Omega_DE_0 = Omega_DE_arr[-1]
    Omega_m_0 = Omega_m_arr[-1]

    # Luminosity distance from H(z) (numerical trapezoid)
    # d_L(z) = (1+z) * integral_0^z dz'/H(z')
    # We have H vs z but z is decreasing in our arrays; reverse for integration
    z_fwd = z_arr[::-1]
    H_fwd = H_arr[::-1]

    # d_C(z) = integral from 0 to z of c*dz'/H(z')
    # In natural units c=1, d_C in GeV^-1
    dC = np.zeros_like(z_fwd)
    for i in range(1, len(z_fwd)):
        dz = z_fwd[i] - z_fwd[i-1]
        if H_fwd[i] > 0 and H_fwd[i-1] > 0:
            dC[i] = dC[i-1] + dz / (0.5 * (H_fwd[i] + H_fwd[i-1]))

    dL = (1.0 + z_fwd) * dC

    # Compute w_a via CPL fit: w(a) = w_0 + w_a (1-a)
    # Use values near z ~ 0.5 to 1.0 for fit
    mask_fit = (z_fwd > 0.1) & (z_fwd < 1.5)
    w_fwd = w_arr[::-1]
    if np.sum(mask_fit) > 10:
        a_fit = 1.0 / (1.0 + z_fwd[mask_fit])
        w_fit = w_fwd[mask_fit]
        # Linear fit: w = w_0_fit + w_a_fit * (1-a)
        X = np.column_stack([np.ones_like(a_fit), 1.0 - a_fit])
        coeffs = np.linalg.lstsq(X, w_fit, rcond=None)[0]
        w0_fit = coeffs[0]
        wa_fit = coeffs[1]
    else:
        w0_fit = w_0
        wa_fit = 0.0

    return {
        'success': True,
        'H0_kms': H0_kms,
        'h': h,
        'H0_GeV': H0_GeV,
        'theta_0': theta_0,
        'theta_0_deg': np.degrees(theta_0),
        'w_0': w_0,
        'w0_cpl': w0_fit,
        'wa_cpl': wa_fit,
        'Omega_DE': Omega_DE_0,
        'Omega_m': Omega_m_0,
        'm_sigma': m_sigma,
        'm_sigma_over_H0': m_sigma / H0_GeV if H0_GeV > 0 else 0,
        # Arrays for plotting / analysis
        'z': z_fwd,
        'H_GeV': H_fwd,
        'H_kms': H_fwd / H_100_GEV * 100.0,
        'w': w_fwd,
        'Omega_DE_z': Omega_DE_arr[::-1],
        'Omega_m_z': Omega_m_arr[::-1],
        'theta_z': theta_arr[::-1],
        'd_L_GeV': dL,
        # Input params
        'Lambda_d_meV': Lambda_d_eV,
        'theta_i': theta_i,
        'f_over_MPl': f_over_MPl,
    }


# =============================================================================
# Main: run default + parameter scan
# =============================================================================
if __name__ == '__main__':
    print("=" * 78)
    print("  BACKGROUND COSMOLOGY: Friedmann + Klein-Gordon with V_A4")
    print("=" * 78)

    # ─── PART 1: Reference run with V_A4 parameters from veff_a4_minimum.py ───
    print(f"\n{'='*78}")
    print("  PART 1: REFERENCE RUN")
    print(f"{'='*78}")

    # From veff_a4_minimum.py: f = 1.81 M_Pl gives m_sigma = H_0
    # theta_i = 2.0 rad (from hunt_H0 analysis, O(1) natural)
    ref = solve_background(
        Lambda_d_eV=2.0,      # 2 meV
        theta_i=2.0,          # rad
        f_over_MPl=1.81,      # from V_A4 analysis
    )

    if ref['success']:
        print(f"\n  Inputs:")
        print(f"    Lambda_d = {ref['Lambda_d_meV']:.1f} meV")
        print(f"    theta_i  = {ref['theta_i']:.2f} rad ({np.degrees(ref['theta_i']):.1f} deg)")
        print(f"    f/M_Pl   = {ref['f_over_MPl']:.3f}")
        print(f"    m_sigma  = {ref['m_sigma']:.3e} GeV")
        print(f"    m_sigma/H0 = {ref['m_sigma_over_H0']:.2f}")
        print(f"\n  ┌────────────────────────────────────────────────┐")
        print(f"  |  H0     = {ref['H0_kms']:7.2f} km/s/Mpc                |")
        print(f"  |  w_0    = {ref['w_0']:.6f}                        |")
        print(f"  |  w0_CPL = {ref['w0_cpl']:.4f},  wa_CPL = {ref['wa_cpl']:.4f}      |")
        print(f"  |  Omega_DE = {ref['Omega_DE']:.4f}                       |")
        print(f"  |  Omega_m  = {ref['Omega_m']:.4f}                       |")
        print(f"  |  theta_0  = {ref['theta_0_deg']:.4f} deg                 |")
        print(f"  └────────────────────────────────────────────────┘")
        dp = ref['H0_kms'] - H0_PLANCK_KMS
        print(f"\n  vs Planck: Delta = {dp:+.2f} ({dp/H0_PLANCK_KMS*100:+.2f}%)")
        print(f"  vs SH0ES: Delta = {ref['H0_kms'] - 73.04:+.2f} ({(ref['H0_kms']-73.04)/73.04*100:+.2f}%)")
    else:
        print(f"  FAILED: {ref['message']}")

    # ─── PART 2: theta_i scan at fixed Lambda_d, f ───
    print(f"\n{'='*78}")
    print("  PART 2: SCAN theta_i  (Lambda_d = 2 meV, f = 1.81 M_Pl)")
    print(f"{'='*78}")

    theta_i_vals = np.linspace(0.3, 2.0, 18)
    print(f"\n  {'theta_i':>8} {'theta_i_deg':>10} | {'H0':>8} {'w_0':>8} {'w0_CPL':>8} {'wa_CPL':>8} {'Omega_DE':>8} {'theta_0':>8}")
    print(f"  {'─'*8} {'─'*10} | {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    best_H0_diff = 999
    best_theta_i = None

    for th_i in theta_i_vals:
        r = solve_background(Lambda_d_eV=2.0, theta_i=th_i, f_over_MPl=1.81)
        if r['success'] and r['H0_kms'] > 0:
            tag = ""
            diff = abs(r['H0_kms'] - H0_PLANCK_KMS)
            if diff < best_H0_diff:
                best_H0_diff = diff
                best_theta_i = th_i
            if diff < 2:
                tag = " <-- PLANCK"
            if abs(r['H0_kms'] - 73.04) < 2:
                tag = " <-- SH0ES"
            print(f"  {th_i:8.3f} {np.degrees(th_i):10.2f} | {r['H0_kms']:8.2f} {r['w_0']:8.4f} "
                  f"{r['w0_cpl']:8.4f} {r['wa_cpl']:8.4f} {r['Omega_DE']:8.4f} {r['theta_0_deg']:8.3f}{tag}")

    if best_theta_i:
        print(f"\n  Best match to Planck: theta_i = {best_theta_i:.3f} rad ({np.degrees(best_theta_i):.1f} deg)")
        print(f"  |H0 - 67.4| = {best_H0_diff:.2f}")

    # ─── PART 3: Lambda_d scan ───
    print(f"\n{'='*78}")
    print("  PART 3: SCAN Lambda_d  (theta_i = 2.0, f = 1.81 M_Pl)")
    print(f"{'='*78}")

    Ld_vals = np.logspace(np.log10(0.5), np.log10(10.0), 20)
    print(f"\n  {'Ld_meV':>8} | {'H0':>8} {'w_0':>8} {'Omega_DE':>8} {'m_s/H0':>8}")
    print(f"  {'─'*8} | {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for Ld in Ld_vals:
        r = solve_background(Lambda_d_eV=Ld, theta_i=2.0, f_over_MPl=1.81)
        if r['success'] and r['H0_kms'] > 0:
            tag = ""
            if abs(r['H0_kms'] - H0_PLANCK_KMS) < 2:
                tag = " <--"
            print(f"  {Ld:8.3f} | {r['H0_kms']:8.2f} {r['w_0']:8.4f} {r['Omega_DE']:8.4f} {r['m_sigma_over_H0']:8.2f}{tag}")

    # ─── PART 4: f scan ───
    print(f"\n{'='*78}")
    print("  PART 4: SCAN f/M_Pl  (Lambda_d = 2 meV, theta_i = 2.0)")
    print(f"{'='*78}")

    f_vals = np.logspace(np.log10(0.3), np.log10(20.0), 20)
    print(f"\n  {'f/MPl':>8} | {'H0':>8} {'w_0':>8} {'Omega_DE':>8} {'m_s/H0':>8}")
    print(f"  {'─'*8} | {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for fv in f_vals:
        r = solve_background(Lambda_d_eV=2.0, theta_i=2.0, f_over_MPl=fv)
        if r['success'] and r['H0_kms'] > 0:
            tag = ""
            if abs(r['H0_kms'] - H0_PLANCK_KMS) < 2:
                tag = " <--"
            print(f"  {fv:8.3f} | {r['H0_kms']:8.2f} {r['w_0']:8.4f} {r['Omega_DE']:8.4f} {r['m_sigma_over_H0']:8.2f}{tag}")

    # ─── PART 5: 2D scan — find best (Lambda_d, theta_i) for Planck H0 ───
    print(f"\n{'='*78}")
    print("  PART 5: 2D SCAN (Lambda_d, theta_i) with f = 1.81 M_Pl")
    print("  TARGET: H0 = 67.4, Omega_DE ~ 0.69, w ~ -1")
    print(f"{'='*78}")

    Ld_2d = np.linspace(1.0, 5.0, 20)
    th_2d = np.linspace(0.3, 2.09, 20)  # up to ~120 deg

    best_score = 999
    best_params = None

    print(f"\n  Best candidates (|H0-67.4| < 5 AND Omega_DE > 0.5):")
    print(f"  {'Ld_meV':>8} {'theta_i':>8} | {'H0':>8} {'w_0':>8} {'w0_CPL':>8} {'wa_CPL':>8} {'Omega_DE':>8}")
    print(f"  {'─'*8} {'─'*8} | {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for Ld in Ld_2d:
        for th in th_2d:
            r = solve_background(Lambda_d_eV=Ld, theta_i=th, f_over_MPl=1.81)
            if not r['success'] or r['H0_kms'] <= 0:
                continue
            dH = abs(r['H0_kms'] - H0_PLANCK_KMS)
            dOm = abs(r['Omega_DE'] - 0.69)
            if dH < 5 and r['Omega_DE'] > 0.5:
                score = dH + 10*dOm
                tag = " ***" if score < best_score else ""
                if score < best_score:
                    best_score = score
                    best_params = (Ld, th, r)
                print(f"  {Ld:8.2f} {th:8.3f} | {r['H0_kms']:8.2f} {r['w_0']:8.4f} "
                      f"{r['w0_cpl']:8.4f} {r['wa_cpl']:8.4f} {r['Omega_DE']:8.4f}{tag}")

    if best_params:
        Ld_b, th_b, rb = best_params
        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  |  BEST FIT:                                          |")
        print(f"  |    Lambda_d = {Ld_b:.2f} meV                              |")
        print(f"  |    theta_i  = {th_b:.3f} rad ({np.degrees(th_b):.1f} deg)                |")
        print(f"  |    f/M_Pl   = 1.81                                   |")
        print(f"  |                                                      |")
        print(f"  |    H0       = {rb['H0_kms']:.2f} km/s/Mpc                    |")
        print(f"  |    Omega_DE = {rb['Omega_DE']:.4f}                            |")
        print(f"  |    w_0      = {rb['w_0']:.4f}                             |")
        print(f"  |    w0_CPL   = {rb['w0_cpl']:.4f}                             |")
        print(f"  |    wa_CPL   = {rb['wa_cpl']:.4f}                             |")
        print(f"  |    theta_0  = {rb['theta_0_deg']:.2f} deg (rolling toward {np.degrees(THETA_RELIC):.2f}) |")
        print(f"  └─────────────────────────────────────────────────────┘")

        # DESI comparison
        print(f"\n  DESI DR2 comparison:")
        print(f"    DESI:  w0 = -0.83 +/- 0.06,  wa = -0.75 +/- 0.25")
        print(f"    Ours:  w0 = {rb['w0_cpl']:.3f},       wa = {rb['wa_cpl']:.3f}")
        dw0 = abs(rb['w0_cpl'] - (-0.83)) / 0.06
        dwa = abs(rb['wa_cpl'] - (-0.75)) / 0.25
        print(f"    Tension: {dw0:.1f} sigma (w0), {dwa:.1f} sigma (wa)")
    else:
        print(f"\n  No candidates found matching Planck H0.")

    # ─── PART 6: Extended 2D with variable f ───
    print(f"\n{'='*78}")
    print("  PART 6: 3D COARSE SCAN (Lambda_d, theta_i, f)")
    print(f"{'='*78}")

    f_3d = [0.5, 1.0, 1.81, 3.0, 5.0, 10.0]
    Ld_3d = np.linspace(1.0, 6.0, 12)
    th_3d = np.linspace(0.5, 2.09, 12)

    best3_score = 999
    best3_params = None

    print(f"\n  {'f/MPl':>6} {'Ld':>6} {'th_i':>6} | {'H0':>7} {'w0':>7} {'w0c':>7} {'wac':>7} {'OmDE':>6}")
    print(f"  {'─'*6} {'─'*6} {'─'*6} | {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6}")

    for fv in f_3d:
        for Ld in Ld_3d:
            for th in th_3d:
                r = solve_background(Lambda_d_eV=Ld, theta_i=th, f_over_MPl=fv)
                if not r['success'] or r['H0_kms'] <= 0:
                    continue
                dH = abs(r['H0_kms'] - H0_PLANCK_KMS)
                dOm = abs(r['Omega_DE'] - 0.69)
                score = dH + 10*dOm + abs(r['w_0'] + 1)*5
                if dH < 3 and r['Omega_DE'] > 0.55:
                    tag = ""
                    if score < best3_score:
                        best3_score = score
                        best3_params = (fv, Ld, th, r)
                        tag = " ***"
                    print(f"  {fv:6.2f} {Ld:6.2f} {th:6.3f} | {r['H0_kms']:7.2f} {r['w_0']:7.4f} "
                          f"{r['w0_cpl']:7.4f} {r['wa_cpl']:7.4f} {r['Omega_DE']:6.4f}{tag}")

    if best3_params:
        fv_b, Ld_b, th_b, rb = best3_params
        print(f"\n  ╔═══════════════════════════════════════════════════════╗")
        print(f"  ║  GLOBAL BEST (3D scan)                                ║")
        print(f"  ║    Lambda_d = {Ld_b:.2f} meV                               ║")
        print(f"  ║    theta_i  = {th_b:.3f} rad ({np.degrees(th_b):.1f} deg)                 ║")
        print(f"  ║    f/M_Pl   = {fv_b:.2f}                                   ║")
        print(f"  ║                                                        ║")
        print(f"  ║    H0       = {rb['H0_kms']:.2f} km/s/Mpc                     ║")
        print(f"  ║    Omega_DE = {rb['Omega_DE']:.4f}                             ║")
        print(f"  ║    w_0      = {rb['w_0']:.4f}                              ║")
        print(f"  ║    w0_CPL   = {rb['w0_cpl']:.4f}                              ║")
        print(f"  ║    wa_CPL   = {rb['wa_cpl']:.4f}                              ║")
        print(f"  ╚═══════════════════════════════════════════════════════╝")
    else:
        print(f"\n  No candidates found in 3D scan.")

    # ─── SUMMARY ───
    print(f"\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}")
    print(f"""
  Potential: V_A4(theta) = -(39/5) Lambda_d^4 cos(theta) + Lambda_d^4 cos(3 theta)
  Shifted:   V_DE(theta) = V_A4(theta) - V_A4(theta_relic)  [V=0 at minimum]
  Minimum:   theta_relic = 18.43 deg (from A4 CG coefficients, sin^2 = 1/10)

  KEY DIFFERENCE from layer8_cosmic_ode.py:
    OLD: V(sigma) = Lambda_d^4 (1 - cos(sigma/f))   [simple cosine]
    NEW: V_A4(theta) with cos(theta) + cos(3theta)   [A4 discrete symmetry]

  The A4 potential has:
    - Steeper walls (cos(3theta) adds harmonics)
    - Minimum at theta_relic = 18.43 deg (not theta = 0)
    - Barrier at theta = 0 (Z3 symmetry point)
""")
