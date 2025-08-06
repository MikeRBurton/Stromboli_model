#!/usr/bin/env python3
"""
coupled_all_driver.py
Fully coupled deep chamber + conduit + shallow dike model
– daily t_eval, diagnostic recalculation, and Qsink = α|Qa|.
"""

from __future__ import annotations
import math, numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── local modules (re‑upload if they have expired) ───────────────────────────
from params                import Params
from conduit_exchange_flux import core_annulus_flux
import conduit_thermal     as ct
import shallow_reservoir   as sr
import phi, viscosity, latent
# ─────────────────────────────────────────────────────────────────────────────

# ── configuration ------------------------------------------------------------
P = Params()                             # parameter container

run_days   = 20_000.0                    # total runtime (days)
t_end      = run_days * 86_400           # seconds
t_step     = 99_500  * 86_400            # perturbation start (sec)

# – initial state vector
y0 = [
    P.T_core0,
    P.T_ann0,
    P.dike_T0,
    P.rho_d * P.dike_l * P.dike_w * P.dike_h0,
    P.P_deep0,
]

# ── RHS function -------------------------------------------------------------
def rhs_full(t: float, y: np.ndarray) -> list[float]:
    T_core, T_ann, T_dike, M_dike, P_deep = y

    # viscosities
    mu_c = viscosity.mu_comp(T_core, phi.phi(T_core), P.h2o_wt,
                             model=P.visc_law)
    mu_a = viscosity.mu_comp(T_ann,  phi.phi(T_ann),  P.h2o_wt,
                             model=P.visc_law)

    # pressures
    P_dike = (M_dike / (P.rho_d * P.dike_l * P.dike_w)) * P.rho_d * P.g

    # conduit exchange fluxes  (Qc up +,  Qa down –)
    Qa, Qc = core_annulus_flux(P_dike, P_deep, mu_c, mu_a,
                               P.rho_c, P.rho_a,
                               delta=P.delta_fixed, L=P.L, R=P.R, g=P.g)

    # shallow‑dike evolution
    H_core_out = latent.cp_eff(T_core) * T_core
    dT_dike, dM_dike, *_ = sr.rhs(t, T_dike, M_dike,
                                  Qc=Qc, Qa=Qa, H_in=H_core_out, p=P)

    # conduit thermal (for latent heat only; fluxes re‑computed later)
    dT_core, dT_ann, *_ = ct.rhs_thermal(
        t, T_core, T_ann, Qc, Qa, P,
        T_in_core=P.T_core0, T_in_ann=T_dike)

    # deep‑chamber volume balance with dense‑sink term, Qa is -ve as downward flux
    # so needs a change in sign as =ve flux into deep chamber
    Qsink = P.alpha_sink * Qa
    Vdot = (P.Q_in - Qc) / P.rho_c + (-(Qa - Qsink)) / P.rho_a

    P_exc  = P_deep - P.P_litho
    dPdeep = Vdot / P.Ce_wall - P_exc / P.tau_wall

    # one‑time perturbation example (increase Q_in by 50 %)
    if t >= t_step and not hasattr(rhs_full, "pert_done"):
        P.dike_area *= 3
        rhs_full.pert_done = True

    return [dT_core, dT_ann, dT_dike, dM_dike, dPdeep]

# ── daily evaluation grid ----------------------------------------------------
t_eval = np.arange(0.0, t_end + 86_400, 86_400)   # seconds (day‑step)

print("⇢  integrating …")
sol = solve_ivp(rhs_full, (0, t_end), y0,
                method='BDF', t_eval=t_eval,
                atol=1e-6, rtol=1e-6)

# ── unpack state arrays (already daily) --------------------------------------
t_d      = sol.t / 86_400                         # days
T_core, T_ann, T_dike, M_dike, P_deep = sol.y
P_dike   = (M_dike / (P.rho_d * P.dike_l * P.dike_w)) * P.rho_d * P.g
V_dike   = M_dike / P.rho_d
H_core   = T_core - 273.15;  H_ann = T_ann - 273.15
P_ext    = P_deep - P_dike
dike_height = P_dike / (P.rho_d * P.g)

# ── re‑compute daily diagnostics (Qc, Qa, heat fluxes) -----------------------
n = t_d.size
Qc_sol   = np.empty(n)
Qa_sol   = np.empty(n)
q_ca_sol = np.empty(n)
q_aw_sol = np.empty(n)

for k in range(n):
    Tc, Ta, Td   = T_core[k], T_ann[k], T_dike[k]
    Md, Pd       = M_dike[k], P_deep[k]
    P_dk         = P_dike[k]

    mu_c = viscosity.mu_comp(Tc, phi.phi(Tc), P.h2o_wt, model=P.visc_law)
    mu_a = viscosity.mu_comp(Ta, phi.phi(Ta), P.h2o_wt, model=P.visc_law)

    Qa, Qc = core_annulus_flux(P_dk, Pd, mu_c, mu_a,
                               P.rho_c, P.rho_a,
                               delta=P.delta_fixed, L=P.L, R=P.R, g=P.g)
    Qa_sol[k], Qc_sol[k] = Qa, Qc

    _, _, q_ca, q_aw = ct.rhs_thermal(
        0.0, Tc, Ta, Qc, Qa, P,
        T_in_core=P.T_core0, T_in_ann=Td)
    q_ca_sol[k], q_aw_sol[k] = q_ca, q_aw

Uc_sol = Qc_sol / (P.rho_c * math.pi * (P.R*P.delta_fixed)**2)
Ua_sol =-Qa_sol / (P.rho_a * math.pi * (P.R**2 - (P.R*P.delta_fixed)**2))
phi_a  = phi.phi(T_ann)
mu_a   = viscosity.mu_comp(T_ann, phi_a, P.h2o_wt, model=P.visc_law)
Vdot = (P.Q_in - Qc_sol) / P.rho_c + (-(Qa_sol - Qa_sol*P.alpha_sink)) / P.rho_a
# ── optional: show only last YEAR days after perturbation -------------------
if hasattr(rhs_full, "pert_done"):
    OFFSET = 2000.0
    mask  = t_d >= (t_d[-1] - OFFSET)
    t_plot = t_d[mask] - t_d[mask][0]

    def keep(x): return x[mask]
    Qc_sol, Qa_sol  = keep(Qc_sol), keep(Qa_sol)
    q_ca_sol, q_aw_sol = keep(q_ca_sol), keep(q_aw_sol)
    Uc_sol, Ua_sol  = keep(Uc_sol), keep(Ua_sol)
    P_deep, P_ext   = keep(P_deep), keep(P_ext)
    V_dike, dike_height = keep(V_dike), keep(dike_height)
    Vdot          = keep(Vdot)
    T_core, T_ann, T_dike = keep(T_core), keep(T_ann), keep(T_dike)
    H_core, H_ann   = keep(H_core), keep(H_ann)
    M_dike          = keep(M_dike)
    phi_a, mu_a     = keep(phi_a), keep(mu_a)
else:
    t_plot = t_d

# ── plotting ---------------------------------------------------------------
fig, ax = plt.subplots(3, 6, figsize=(18, 9)); ax = ax.flat
ax[0].plot(t_plot, q_ca_sol/1e6);              ax[0].set_title("Core→Ann (MW)")
ax[1].plot(t_plot, q_aw_sol/1e6);              ax[1].set_title("Ann→Wall (MW)")
ax[2].plot(t_plot, Uc_sol); ax[2].plot(t_plot, abs(Ua_sol))
ax[2].legend(['Uc', '|Ua|']);                  ax[2].set_title("Velocities (m s⁻¹)")
ax[3].plot(t_plot, P_deep/1e6);                ax[3].set_title("P_deep (MPa)")
ax[4].plot(t_plot, V_dike/1e6);                ax[4].set_title("dike V (Mm³)")
ax[5].plot(t_plot, (q_ca_sol+q_aw_sol)/1e6);   ax[5].set_title("Net conduit power")
# row 2
ax[6].plot(t_plot, H_core);                    ax[6].set_title("Core T (°C)")
ax[7].plot(t_plot, H_ann , 'orange');          ax[7].set_title("Ann T (°C)")
ax[8].plot(t_plot, T_core - T_ann);            ax[8].set_title("ΔT core‑ann (K)")
ax[9].plot(t_plot, Qc_sol);                    ax[9].set_title("Qc (kg s⁻¹)")
ax[10].plot(t_plot, H_ann , 'orange');         ax[10].set_title("dike T (°C)")
ax[11].plot(t_plot, P_ext/1e6);                ax[11].set_title("P_ext (MPa)")
# row 3
ax[12].plot(t_plot, Vdot);                ax[12].set_title("Volumetric flux into deep (m³ s⁻¹)")
ax[13].plot(t_plot, mu_a);                     ax[13].set_title("Ann μ (Pa s)")
ax[14].plot(t_plot, Qc_sol*0.004*86.4);        ax[14].set_title("SO₂ flux (t d⁻¹)")
ax[15].plot(t_plot, Qa_sol);                   ax[15].set_title("Qa (kg s⁻¹)")
ax[16].plot(t_plot, M_dike/(Qc_sol*365*86400));ax[16].set_title("dike residence (yr)")
ax[17].plot(t_plot, dike_height);              ax[17].set_title("Dike height (m)")

for a in ax:
    a.set_xlabel("Day"); a.grid(ls=":")

fig.suptitle("Fully coupled model – mass, heat & pressure", fontweight='bold')
fig.tight_layout(); plt.show()
