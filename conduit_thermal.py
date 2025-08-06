from __future__ import annotations
import numpy as np
import latent, viscosity, phi      # your helper modules
from params import Params

def rhs_thermal(
    t: float,
    T_c: float,
    T_a: float,
    Qc: float,
    Qa: float,
    p: Params,
    *,
    T_in_core: float | None = None,
    T_in_ann : float | None = None,
) -> tuple[float, float, float, float]:
    """
    Pure thermal RHS for conduit core & annulus.

    Returns
    -------
    dT_c_dt  – core temperature rate (K s‑1)
    dT_a_dt  – annulus temperature rate (K s‑1)
    q_ca     – conductive heat rate core→annulus (W, + if core hotter)
    q_aw     – conductive heat rate annulus→wall  (W, + if annulus hotter)
    """

    # ------------- incoming temperatures ---------------------
    Tin_c = T_in_core if T_in_core is not None else p.T_in_core
    Tin_a = T_in_ann  if T_in_ann  is not None else p.T_in_ann

    # ------------- effective heat capacities ----------------
    cp_c = latent.cp_eff(T_c)          # J kg‑1 K‑1
    cp_a = latent.cp_eff(T_a)

    # ------------- conductive heat (Fourier) ----------------
    q_ca = p.h_c * p.SA_core * (T_c - T_a)          # W core to annulus
    q_aw = p.h_w * p.SA_ann * (T_a - p.T_wall)     # W annulus to wall

    # ------------- viscous heating Φ (optional) -------------
    visc_c = visc_a = 0.0
    if p.viscous_heating:
        mu_c = viscosity.mu_comp(T_c, phi.phi(T_c), 3.0, model=p.visc_law)
        mu_a = viscosity.mu_comp(T_a, phi.phi(T_a), 1.0, model=p.visc_law)

        # convert mass‑flux → volume‑flux
        Qv_c = Qc / p.rho_c               # m³ s‑1   (upward, sign preserved)
        Qv_a = Qa / p.rho_a               # m³ s‑1   (downward, sign preserved)

        visc_c = 8 * mu_c * Qv_c**2 * p.L / (np.pi * p.Rc**4)
        Ra4 = np.pi * (p.R**4 - p.Rc**4)
        visc_a = 8 * mu_a * Qv_a**2 * p.L / Ra4

    # ------------- advective replacement --------------------
    #   f = (mass replaced per second) / (mass in reservoir)
    f_c = abs(Qc) / (p.m_c)              # 1/s
    f_a = abs(Qa) / (p.m_a) if Qa else 0.0

    # ------------- temperature ODEs -------------------------
    dT_c_dt = (-q_ca + visc_c)/(p.m_c * cp_c) + f_c * (Tin_c - T_c)
    dT_a_dt = (+q_ca - q_aw + visc_a)/(p.m_a * cp_a) + f_a * (Tin_a - T_a)

    return dT_c_dt, dT_a_dt, q_ca, q_aw
