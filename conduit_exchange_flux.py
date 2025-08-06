"""
conduit_exchange_flux.py
========================
Compute signed dimensional mass fluxes for core–annulus
exchange flow, suitable for direct use inside coupled_all_driver.

Requirements
------------
* fluid_model_dissipation.py must be importable.

Author: <you>
Last updated: <today>
"""

from __future__ import annotations
import numpy as np
import fluid_model_dissipation as fmd


def core_annulus_flux(
        P_top      : float,
        P_bottom   : float,
        mu_c       : float,
        mu_a       : float,
        rho_c      : float,
        rho_a      : float,
        *,
        delta: float = 0.30,
        L    : float = 8_000.0,
        R    : float = 1.60,
        g    : float = 9.81
) -> tuple[float, float]:
    """
    Dimensional mass fluxes (kg s⁻¹) for a concentric core (up) &
    annulus (down) magma column.

    Parameters
    ----------
    P_top, P_bottom : float
        Absolute pressures at the conduit top and base (Pa).
    mu_c, mu_a      : float
        Dynamic viscosities of core and annulus (Pa s).
    rho_c, rho_a    : float
        Densities of core and annulus (kg m⁻³).
    delta           : float, default 0.30
        Fixed non‑dimensional core radius r_c/R.
    L               : float, default 8 000.0
        Conduit length (m).
    R               : float, default 1.60
        Conduit outer radius (m).
    g               : float, default 9.81
        Gravity (m s⁻²).

    Returns
    -------
    Qa, Qc : float, float
        *Qa*  – annulus mass flux (kg s⁻¹, **negative** = downward)  
        *Qc*  – core    mass flux (kg s⁻¹, **positive** = upward)

    Notes
    -----
    • The pressure gradient is defined as  dP/dz = (P_top−P_bottom)/L,  
      hence positive when pressure decreases with depth.  
    • The redimensionalisation follows
        α = R⁴ (ρ_a−ρ_c) g / μ_a,
      exactly as in map_fixed_delta.py.
    """
    # ---- dimensional → nondimensional pressure --------------------------
    dpdz = (P_top - P_bottom) / L                 # Pa m⁻¹   (usually >0)
    P_nd = (dpdz + rho_a * g) / (g * (rho_a - rho_c))

    # ---- viscosity ratio -------------------------------------------------
    M = mu_a / mu_c

    # ---- nondimensional volume fluxes ------------------------------------
    qc_nd, qa_nd = fmd.nondim_conduit_volume_flux(P_nd, M, delta=delta)

    # ---- redimensional to signed mass fluxes -----------------------------
    alpha = (R ** 4) * (rho_a - rho_c) * g / mu_a
    Qc =  rho_c * alpha * qc_nd          # up‑flow  (+)
    Qa =  rho_a * alpha * qa_nd          # down‑flow (−)

    return Qa, Qc


# -------------------------------------------------------------------------
# Simple self‑test (runs when file executed directly)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example inputs (same numbers as your latest map)
    P_top, P_bottom = 20e6, 200e6         # Pa
    mu_c, mu_a      = 200.0, 20_000.0     # Pa s  (M = 100)
    rho_c, rho_a    = 2_200.0, 2_700.0    # kg m⁻³

    Qa, Qc = core_annulus_flux(P_top, P_bottom,
                               mu_c, mu_a, rho_c, rho_a)

    print(f"Qa = {Qa:8.1f} kg s⁻¹   (down ≡ –)")
    print(f"Qc = {Qc:8.1f} kg s⁻¹   (up)")
    print(f"Qnet = {Qc+Qa:8.1f} kg s⁻¹")
