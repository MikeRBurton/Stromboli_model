#!/usr/bin/env python3
"""
shallow_reservoir.py
--------------------
0‑D mass–energy balance for the shallow dike‑like reservoir.

State vector      :  y = [T_sh  (K),  M_sh  (kg)]
Caller must pass  :  Qc  (kg s⁻¹)   core mass flux (positive is into dike)
                    Qa (kg s⁻¹)    annulus mass flux (negative is out of dike)
                    H_in  (J kg⁻¹)    enthalpy of the incoming magma
"""

from __future__ import annotations
import numpy as np
import latent
from params import Params


# --------------------------------------------------------------------------
def rhs(t: float, T: float, M: float,
        *, Qc: float, Qa: float, H_in: float, p: Params) -> tuple[float, float]:
    """
    Returns dT/dt [K s⁻¹], dM/dt [kg s⁻¹] for the shallow reservoir.
    """

    # ----------------------------------- mass balance
    m_in  = Qc  # kg s⁻¹ (positive is into dike)
    m_out = -Qa # kg s⁻¹ (negative is out of dike)
    dM_dt     = m_in - m_out                  # kg s⁻¹

    # ----------------------------------- energy balance
    #print(f"Time={t:<.2f}, dM/dt in from core={m_dot_in:<.2f}, dM/dt out to annulus={m_dot_out:<.2f}, Dike mass={M:<.2f}")
    # ------------------------------------------------------------------ geometry
    # Shallow reservoir area changes with the instantaneous dike height
    #height = M / (p.rho_d * p.dike_l * p.dike_w)         # m using dike density rho_d
    #A_sh = 2 * (p.dike_l * p.dike_w) + 2 * (p.dike_l * height) + 2 * (height * p.dike_w)    # area of shallow reservoir m²
    #A_sh = p.dike_area  # m² (fixed dike area)
    # ------------------------------------------------------------------ conduction to host rock
    h_w   = p.k_r_d / p.delta_w_d                           # W m⁻² K⁻¹
    q_cond = h_w * p.dike_area * (T - p.dike_wall_T)                        # W ( = J s⁻¹)
    
    # ------------------------------------------------------------------ latent‑aware heat capacity
    cp_eff = latent.cp_eff(T)                                     # J kg⁻¹ K⁻¹

    # ------------------------------------------------------------------ enthalpies (J kg⁻¹)
    H = cp_eff * T  # J kg⁻¹ (enthalpy of the dike magma)

     # energy fluxes (W = J s‑1)
    E_dot = m_in*H_in                          # advected in
    E_dot -= m_out*H                           # advected out
    E_dot -= q_cond                            # conduction loss

    # temperature ODE   (from  dE = H dM + M cp dT )
    dT_dt = (E_dot - H*dM_dt) / (M * cp_eff)

    #net power
    net_power = E_dot - H * dM_dt
    
    #print('Time=',t,' Power in from Qc',p.rho_c*Qc*H_in,' Power in from Qc',
    #      p.rho_a*Q_out*H_out,' Power out from conduction',q_cond,' Net power=',net_power)

    return dT_dt, dM_dt, q_cond, net_power
