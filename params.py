#!/usr/bin/env python3
"""
params.py ─ single source‑of‑truth for every constant.

Edit here → propagation to all sub‑modules.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

@dataclass
class Params:


    # ------------------------- conduit geometry --------------------------------
    R  : float = 1.5             # m
    delta_fixed : float = 0.3     
    Rc : float = field(init=False)  # m (fixed core radius)
    L  : float = 7_000.0          # m

    # ------------------------- initial conduit temps --------------------------
    T_core0 : float = 1130.0 + 273.15        # K
    T_ann0  : float = 1112.0 + 273.15        # K
    T_in_core : float = 1130.0 + 273.15      # K (deep‑inlet)
    T_in_ann  : float = 1112.0 + 273.15      # K (shallow return)

    # ------------------------- deep chamber -----------------------------------
    P_deep0 = 180e6     # Pa  (200 MPa)
    P_litho = 180e6     # Pa  (200 MPa, lithostatic pressure)
    Q_in  : float = 600           # kg s‑1 magma supply from mantle to deep reservoir
    V_d   : float = 3.0e8         # m³
    beta_m : float = 0.9e-10      # melt compressibility
    beta_cr: float = 1.0e-11      # host‑rock compressibility
    alpha_sink: float = 1.0       # fraction of |Qa| stored in ductile floor

    # ------------------------- deep‑chamber wall rheology --------------------------
    K_wall  : float = 5.0e10        # Pa   (elastic bulk modulus of chamber wall)
    eta_wall: float = 2.0e16        # Pa·s (Maxwell viscosity of wall rock)
    
    # -------------------- derived (add two more) -------------------------------
    Ce_wall : float = field(init=False)   # compliance  V_d / K
    tau_wall: float = field(init=False)   # Maxwell time eta / K

    # ------------------------- shallow dike ------------------------------------
    dike_w : float = 100.0 #100        # m
    dike_l : float = 200.0 #200       # m
    dike_h0 : float = 100.0       # m produces exactly 20.83 MPa pressure at the dike conduit exit
    dike_area : float = 1.27e5 # m² (fixed dike area)
    dike_T0: float = 1112.0+273.15# K
    k_r_d  : float = 3.0 #3.0          # W m‑1 K‑1
    delta_w_d: float = 1.7      # m (initial conductive halo thickness)
    dike_wall_T: float = 273.15 + 20.0 + 20.0 # K dike wall temperature
    rho_d : float = 2700.0        # kg m‑3 (same as annulus bulk)
    P_dike0: float = dike_h0*rho_d*9.81 # Pa (initial dike pressure, same as Pd0)

    # ------------------------- thermal envelopes -------------------------------
    k_m      : float = 2.0  #2.0      # W m‑1 K‑1 (magma)
    delta_ca : float = 0.3       # m (core ↔ annulus boundary layer)
    k_r      : float = 2.0 #3.0       # W m‑1 K‑1 (wall rock)
    delta_w  : float = 100  #5-20      # m (annulus ↔ wall shell)

    # ------------------------- physical ----------------------------------------
    rho_c : float = 2200.0
    rho_a : float = 2700.0
    g     : float = 9.81
    cp    : float = 1200.0
    h2o_wt : float = 0.43        # h2o wt fract% in core and annulus

    # ------------------------- switches ----------------------------------------
    plug_mu_max  : float = 2e5     # Pa s
    visc_law     : str   = 'Misiti'
    flow_model   : str   = 'MIKE'  # or 'MIKE' or 'CHRIS
    tau_m        : float = 30.0    # s (flux relaxation)
    viscous_heating : bool = False

    # -------------------- derived (filled in __post_init__) --------------------
    area_c: float = field(init=False); area_a: float = field(init=False)
    A_int: float = field(init=False);  A_wall: float = field(init=False)
    h_c  : float = field(init=False);  h_w   : float = field(init=False)
    T_wall: float = field(init=False)
    m_c  : float = field(init=False);  m_a   : float = field(init=False)
    V_c  : float = field(init=False);  V_a   : float = field(init=False)

    # dike – initial values
    A_dike_wall: float = field(init=False)

    def __post_init__(self):
        # --- derived parameters ---------------------------------------------
        # core radius (fixed geometry version)
        self.Rc = self.R * self.delta_fixed
        self.SA_ann  = 2*math.pi * self.R  * self.L # annulus to crust surface area (m²)
        self.SA_core = 2*math.pi * self.Rc * self.L # core to annulus surface area (m²)
        self.V_c    = math.pi * (self.R * self.delta_fixed)**2 * self.L
        self.V_a    = math.pi * self.R**2 * self.L - self.V_c
        self.h_c    = self.k_m / self.delta_ca
        self.h_w    = self.k_r / self.delta_w
        self.T_wall = (25.0/1000.0)*self.L/2 + 273.15+200.0   # K
        self.m_c  = self.rho_c * self.V_c
        self.m_a  = self.rho_a * self.V_a

        # dike wall (perimeter*height; height will evolve)
        self.A_dike_wall = 2*(self.dike_l+self.dike_w)*self.dike_w

            # visco‑elastic wall
        # --- deep‑reservoir wall mechanics  --------------------------------------
        self.Ce_wall  = self.V_d / self.K_wall          #  m³ Pa⁻¹  (Hooke’s law)
        self.tau_wall = self.eta_wall / self.K_wall    #  s