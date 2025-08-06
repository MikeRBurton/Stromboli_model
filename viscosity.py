"""
viscosity.py  — Stromboli basalt melt + crystal viscosity
==========================================================

Includes TWO suspension laws:

1.  "KD"      – Krieger–Dougherty (what we used in v12)
2.  "Misiti"  – empirical fit of Misiti et al. (2005, 2006)
                log10(η/η_m)  =  a·φ + b·φ² + c·φ³     (see constants)

Usage
-----
>>> mu = mu_comp(T_K, phi, H2O_wt=0.3, model="Misiti")
"""
from __future__ import annotations
import numpy as np
import phi                       # crystal fraction module

# ------------------------------------------------------------------
# 1. Giordano et al. (2008) melt viscosity for Stromboli basalt
# ------------------------------------------------------------------
_A, _B, _C = -4.55, 8208.9, 0.794          # composition constants

def melt_viscosity(T_K: float | np.ndarray, H2O_wt: float = 0.3):
    """Basaltic melt viscosity (Pa·s)."""
    delta = -0.313 * H2O_wt                     # water effect
    log_eta = _A + _B / (T_K - _C) + delta
    return 10.0 ** log_eta


# ------------------------------------------------------------------
# 2A. Krieger–Dougherty multiplier
# ------------------------------------------------------------------
PHI_MAX_KD = 0.60      # random‐close packing
KD_EXP     = 2.5 * PHI_MAX_KD

def _multiplier_KD(phi_v):
    phi_v = np.clip(phi_v, 0.0, 0.999*PHI_MAX_KD)
    return (1.0 - phi_v/PHI_MAX_KD) ** -KD_EXP


# ------------------------------------------------------------------
# 2B. Misiti et al.  approach  (plug‑flow fit to Stromboli data)
#     log10(η/η_m) = a φ + b φ² + c φ³
#     Coefficients: Misiti 2005, J. Rheology 49, 651–670  (St212 basalt)
# ------------------------------------------------------------------
_aM, _bM, _cM = 1.40, 5.00, 30.0   # dimensionless, calibrated for φ≤0.4

def _multiplier_Misiti(phi_v):
    phi_v = np.clip(phi_v, 0.0, 0.45)   # formula calibrated ≤0.45
    log10_mult = _aM*phi_v + _bM*phi_v**2 + _cM*phi_v**3
    return 10.0 ** log10_mult


# ------------------------------------------------------------------
# 3. Composite viscosity
# ------------------------------------------------------------------
def mu_comp(T_K: float | np.ndarray,
            phi_v: float | np.ndarray,
            H2O_wt: float = 0.3,
            model: str = "KD") -> np.ndarray:
    """
    Effective viscosity of melt + crystals.

    Parameters
    ----------
    T_K     : temperature [K]
    phi_v   : crystal volume fraction (0–0.6)
    H2O_wt  : dissolved water wt %
    model   : "KD"  or  "Misiti"

    Returns
    -------
    η  [Pa·s]
    """
    eta_m = melt_viscosity(T_K, H2O_wt)

    if model.lower() == "misiti":
        mult = _multiplier_Misiti(phi_v)
    else:                               # defaults to KD
        mult = _multiplier_KD(phi_v)

    return eta_m * mult


# ------------------------------------------------------------------
# 4. Convenience wrapper  (uses phi.phi(T))
# ------------------------------------------------------------------
def mu_from_T(T_K: float | np.ndarray,
              H2O_wt: float = 0.3,
              model: str = "KD") -> np.ndarray:
    """Viscosity using φ(T) from phi.py."""
    return mu_comp(T_K, phi.phi(T_K), H2O_wt, model)


# ------------------------------------------------------------------
# 5. Self‑test
# ------------------------------------------------------------------
if __name__ == "__main__":
    T_list = np.linspace(1000+273.15, 1150+273.15, 20)
    for T in T_list:
        ph = phi.phi(T)
        mu_kd = mu_comp(T, ph, model="KD")
        mu_mi = mu_comp(T, ph, model="Misiti")
        print(f"{T-273.15:6.0f} °C  φ={ph:5.2f}  "
              f"μ_KD={mu_kd*1e-3:9.2e} kPa·s  μ_Misiti={mu_mi:9.2e} Pa·s")
