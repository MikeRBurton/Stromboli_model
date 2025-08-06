# phi.py  – Stromboli basalt crystal fraction 800–1200 °C
# Uses output from EasyMelts Melts V1.2.0 with whole rock
# composition from Landi 2004 St212 
# STR shallow 1200C to 800C just xtal Landi 2004 St212.csv

"""
phi.py  – crystal volume fraction for Stromboli basalt
------------------------------------------------------
Reads the MELTS run (1200–800 °C, 20 MPa) and provides

    phi(T_K)      – volume fraction of crystals (0–0.6)
    dphi_dT(T_K)  – central‑difference slope (K⁻¹)

The CSV must have these columns:
    T (C), P (bar), m (melt), m (solid)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# 1.  Load the CSV, regardless of comma or whitespace format
# ------------------------------------------------------------
def _load_table() -> tuple[np.ndarray, np.ndarray]:
    csv = Path(__file__).with_name(
        "STR shallow 1200C to 800C just xtal Landi 2004 St212.csv")

    # Try comma first; fall back to whitespace
    try:
        df = pd.read_csv(csv, comment="#")            # comma‑separated
    except pd.errors.ParserError:
        df = pd.read_csv(csv, delim_whitespace=True, comment="#")

    # Force numeric dtype; drop rows with NaNs
    for col in ["T (C)", "m (melt)", "m (solid)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["T (C)", "m (melt)", "m (solid)"])

    # Mass → volume fraction
    rho_melt, rho_xtal = 2600.0, 3000.0           # kg m⁻³
    Xs = df["m (solid)"] * 1e-2                   # crystal mass fraction
    Xm = df["m (melt)"]  * 1e-2
    phi_vol = (Xs/rho_xtal) / ((Xs/rho_xtal) + (Xm/rho_melt))

    return df["T (C)"].values, phi_vol.values


# ------------------------------------------------------------
# 2.  Build cubic‑spline interpolator  (800–1200 °C)
# ------------------------------------------------------------
_T_C, _phi_vol = _load_table()
_phi_of_T = interp1d(_T_C, _phi_vol, kind="cubic",
                     fill_value="extrapolate", bounds_error=False)

def phi(T_K: float | np.ndarray) -> np.ndarray:
    """Crystal volume fraction (0–0.98) from 800 °C to 1200 °C."""
    return np.clip(_phi_of_T(T_K - 273.15), 0.0, 1.0)

def dphi_dT(T_K):
    """Central‑difference slope (K⁻¹) for latent‑heat term."""
    h = 1.0
    return (phi(T_K + 0.5*h) - phi(T_K - 0.5*h)) / h


# ------------------------------------------------------------
# 3.  Self‑test
# ------------------------------------------------------------
if __name__ == "__main__":
    for T_C in (1200, 1100, 1000, 900, 800):
        T_K = T_C + 273.15
        print(f"{T_C:4.0f} °C → φ = {phi(T_K):.4f}")
