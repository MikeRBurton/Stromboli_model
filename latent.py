# cp_eff combines both sensible heat and latent heat of crystallisation
# into one parameter which captures the effect of enthalpy release from crystal formation
# so we need to remove more than just sensible heat to reduce temperatures

import numpy as np
from phi import phi, dphi_dT                         # where data from EasyMelts on Stromboli is used

CP_MELT = 1.2e3      # J kg^-1 K^-1
CP_XTAL = 0.9e3      # J kg^-1 K^-1
LATENT  = 3.2e5      # J kg^-1

def cp_eff(T_K: float | np.ndarray) -> np.ndarray:
    """Sensible + latent effective heat capacity (J kg^-1 K^-1)."""
    phi_T = phi(T_K)  # volume fraction of crystals
    dphi_dT_T = dphi_dT(T_K)  # central-difference slope (K⁻¹)

    new_cp_eff = CP_MELT * (1.0 - phi_T) + CP_XTAL * phi_T + LATENT * abs(dphi_dT_T)

    return new_cp_eff