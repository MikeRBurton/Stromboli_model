"# fluid_model_dissipation.py"

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Precalculation of delta_from_p lookup table

# Parameters for the lookup table
delta_min = 1e-10  # Smallest delta to avoid log(0) and division by zero
delta_max = 1 - 1e-10  # Largest delta to avoid division by zero at delta=1
num_points = 1000  # Number of points 

# Generate delta values (excludes 0 and 1 to avoid numerical issues)
delta_array = np.linspace(delta_min, delta_max, num_points)

# Compute P = (2 * delta^2 * log(delta)) / (delta^2 - 1)
denominator = delta_array**2 - 1
P_array = 2 * delta_array**2 * np.log(delta_array) / (delta_array**2 - 1)


def celcius_to_kelvin(celc):
    return celc + 273.15


################################################################################
# Global parameters

g = 9.81 # acceleration due to gravity, m/s^2


               
################################################################################
# Conduit fluid flow model:
#
# Uniaxial core-annular flow in a vertical cylindrical conduit of radius R.
# Core flow has different (lower) viscosity and density than annular flow.
#
# Combines Suckale (2018) equations, but imposing that the velocity at the
# core-annulus interface is zero, rather than their flux condition (2.10)
#
# The logic for zero velocity at the interface is that, in a steady state,
# the type of fluid depends on the reservoir that it came from, which is
# determined exactly by the sign of u. This assumes that all core fluid that
# reaches the top of the conduit is cooled and becomes the denser annulus fluid;
# that is, none is reversed within the conduit or near the conduit end. Likewise
# for dense fluid reaching the bottom of the conduit.
#
#
# Subscripts used: _c = core flow;  _a = annular flow
# (These differ from Suckale, who use _a (ascending) and _d (descending))

# Nondimensional parameters defined:
# pressure drop P = (dp/dz + rho_a g)/(g (rho_a - rho_c)
# viscosity ratio M = mu_a / mu_c

################################################################################

# Returns dimensionless radius of conduit core, delta, as function
# of dimensionless pressure drop P
def delta_from_P(P):
    if (P>=1 or P < 0):
        return 1
    else:
        return np.interp(P, P_array, delta_array, left=0.0, right=1.0)   

################################################################################

# Returns nondimensional volume flux in core and annulus of conduit, as function of P and M
# Output fluxes should be redimensionalised into volume fluxes by multiplying by:
#  R^4 (rho_a - rho_c) g / mu_a
# and converted to mass flux by multiplying by the appropriate density
def nondim_conduit_volume_flux(P, M, delta = None):
    if delta is None:
        delta = delta_from_P(P)
        
    # core flux
    q_c = (np.pi/8) * delta**2 *(-2*P + (M+2*P-M*P)*delta**2 - 4 * delta**2 * np.log(delta))
    # annular flux
    q_a = (np.pi/8)*(4*delta**4*np.log(delta) - (delta**2 - 1)*(delta**2*(P+2)-P))
    return (q_c, q_a)

################################################################################

# Returns nondimensional velocity profiles in core and annulus, given P and M
def nondim_velocity_profile(P, M, delta = None):
    if delta is None:
        delta = delta_from_P(P)

    n_r = 250
    
    # Generate r
    r_c = np.linspace(0, delta, n_r, endpoint = False)
    r_a = np.linspace(delta, 1, n_r)

    u_c = (M*(P-1)/4)*(r_c**2 - delta**2) + (P/4)*(delta**2 - 1) - (delta**2 / 2)*np.log(delta)
    u_a = (P/4)*(r_a**2 - 1) - delta**2*np.log(r_a)/2

    return (r_c, u_c, r_a, u_a)



################################################################################

# Returns 'nondimensional stress' profiles in core and annulus, given P and M
# These are just du/dr (for nondimensional u and r)
def nondim_stress_profile(P, M, delta):
    if delta is None:
        delta = delta_from_P(P)

    n_r = 250
    
    # Generate r
    r_c = np.linspace(0, delta, n_r, endpoint = False)
    r_a = np.linspace(delta, 1, n_r)

    du_c = (M*(P-1)/4)*(2*r_c)
    du_a = (P/2)*r_a - delta**2 / (r_a * 2)

    return (r_c, du_c, r_a, du_a)
    
################################################################################

# Returns dimensional conduit mass fluxes in core and annulus,
# given dimensional:
#    conduit radius R
#    pressure gradient dpdz
#    densities rho_a, rho_c
#    viscosities mu_a, mu_c
#
# Uses global gravity g
def conduit_mass_flux(R, dpdz, rho_a, rho_c, mu_a, mu_c, delta = None):

    # Generate nondimensional parameters
    P = (dpdz + rho_a*g)/(g*(rho_a - rho_c))
    M = mu_a / mu_c

    # Calculate nondimensional volume fluxes 
    q_c, q_a = nondim_conduit_volume_flux(P, M, delta)

    # redimensionalisation volume flux
    dim_volume = R**4 * (rho_a - rho_c)*g / mu_a

    # Convert to dimensional mass fluxes
    qm_c = rho_c * dim_volume * q_c
    qm_a = rho_a * dim_volume * q_a

    return (qm_c, qm_a)

################################################################################

# Returns nondimensional viscous dissipation, as a function of P, M, delta

def nondim_dissipation(P, M, delta):
    return (1/16) * M * (P-1)**2 * delta**4  +  (1/16) * (P * (P - 4 * delta**2 - (P-4)*delta**4) - 4 * delta**4 * np.log(delta))

################################################################################

# Returns dimensional viscous dissipation per unit length of conduit,
# given dimensional:
#    conduit radius R
#    pressure gradient dpdz
#    densities rho_a, rho_c
#    viscosities mu_a, mu_c
#
# and dimensionless delta
# Uses global gravity g

def conduit_dissipation(R, dpdz, rho_a, rho_c, mu_a, mu_c, delta):

    # Generate nondimensional parameters
    P = (dpdz + rho_a*g)/(g*(rho_a - rho_c))
    M = mu_a / mu_c

    # Calculate nondimensional volume fluxes 
    nd_dis = nondim_dissipation(P, M, delta)

    # redimensionalisation power per unit length
    dim_power_per_length = np.pi * R**4 * (rho_a - rho_c)**2 * g**2 / mu_a

    # Convert to dimensional mass fluxes
    return dim_power_per_length * nd_dis


################################################################################

# Returns dimensional conduit velocity profiles
def conduit_velocity_profile(R, dpdz, rho_a, rho_c, mu_a, mu_c, delta = None):
    
    # Generate nondimensional parameters
    P = (dpdz + rho_a*g)/(g*(rho_a - rho_c))
    M = mu_a / mu_c

    # redimensionalisation velocity
    U = (rho_a - rho_c)*g*R**2 / mu_a

    r_c, u_c, r_a, u_a = nondim_velocity_profile(P, M, delta)

    r_c *= R # redimensionalise
    r_a *= R
    u_c *= U
    u_a *= U

    return (r_c, u_c, r_a, u_a)



################################################################################

# Returns dimensional conduit stress profiles
def conduit_stress_profile(R, dpdz, rho_a, rho_c, mu_a, mu_c, delta = None):
    
    # Generate nondimensional parameters
    P = (dpdz + rho_a*g)/(g*(rho_a - rho_c))
    M = mu_a / mu_c

    # redimensionalisation velocity
    U = (rho_a - rho_c)*g*R**2 / mu_a

    r_c, sigma_c, r_a, sigma_a = nondim_stress_profile(P, M, delta)

    r_c *= R # redimensionalise
    r_a *= R
    sigma_c *= (U/R) * mu_c
    sigma_a *= (U/R) * mu_a

    return (r_c, sigma_c, r_a, sigma_a)



################################################################################
# Evaluate fluxes and plot velocity and stress profiles

def plot_profiles():

    R = 1.4
    
    T_c, T_a = 1160, 1130 # celcius
    rho_c, rho_a = 2400, 2700
#    mu_c, mu_a = 500, 1500; old pre

    phi_c = 1#viscosity.phi_of_T(T_c)
    phi_a = 1#viscosity.phi_of_T(T_a)

    mu_c = 1#viscosity.mu_comp(celcius_to_kelvin(T_c), phi_c)
    mu_a = 1#viscosity.mu_comp(celcius_to_kelvin(T_a), phi_a)

    
    # This pressure gradient has been tuned to something close to the value
    # that produces no net mass flux in the conduit
  #  dpdz, delta = -23643, 0.707
   # dpdz, delta = -23681, 0.6
    dpdz, delta = -23900, 0.3    
    # Calculate and display mass fluxes
    # qm_c, qm_a = conduit_mass_flux(R, dpdz, rho_a, rho_c, mu_a, mu_c, delta)
    
    # print('Viscosity in core: ', mu_c, 'Pa s')
    # print('Viscosity in annulus: ', mu_a, 'Pa s')
    # print('Mass flux in core: ', qm_c, 'kg/s')
    # print('Mass flux in annulus: ', qm_a, 'kg/s')
    # print('Total conduit mass flux: ', qm_c + qm_a, 'kg/s')


    # # Calculate and plot velocity and stress profiles
    # r_c, u_c, r_a, u_a = conduit_velocity_profile(R, dpdz, rho_a, rho_c, mu_a, mu_c, delta)
    # r_c, sigma_c, r_a, sigma_a = conduit_stress_profile(R, dpdz, rho_a, rho_c, mu_a, mu_c, delta)

    # fig, ax = plt.subplots(1, 2, figsize=(9,4))

    # ax[0].plot(r_c, u_c, 'b', label='core')
    # ax[0].plot(r_a, u_a, 'r', label='annulus')
    # ax[0].set_xlabel('r [m]');
    # ax[0].set_ylabel('u [m s⁻¹]')


    # ax[1].plot(r_c, sigma_c, 'b', label='core')
    # ax[1].plot(r_a, sigma_a, 'r', label='annulus')
    # ax[1].set_xlabel('r [m]');
    # ax[1].set_ylabel('sigma [kg m⁻¹ s⁻²]')


    # plt.tight_layout()
    # plt.show()

    print(conduit_dissipation(R,dpdz,rho_a,rho_c,mu_a,mu_c,0.2))

    
    
if __name__ == "__main__":
    plot_profiles()
