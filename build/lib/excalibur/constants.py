import scipy.constants as sc

#***** Pressure broadening settings *****#

gamma_0_fixed = 0.07    # If fixed broadening chosen, use this Lorentzian HWHM
n_L_fixed = 0.50        # If fixed broadening chosen, use this temperature exponent

T_ref = 296.0   # Reference temperature for broadening parameters
P_ref = 1.0     # Reference temperature for EXOMOL broadening parameters (bar) - HITRAN conversion from atm already pre-handled

#***** Define physical constants *****#

c = sc.c          # Speed of light (SI) = 299792458.0 m s^-1
kb = sc.k         # Boltzmann constant (SI) = 1.38064852e-23 m^2 kg s^-2 K^-1
h = sc.h          # Planck's constant (SI) = 6.62607004e-34 m^2 kg s^-1
m_e = sc.m_e      # Electron mass (SIT) = 9.10938356e-31 kg
c2 = h*c/kb       # Second radiation constant (SI) = 0.0143877736 m K  
c2 *= 100.0       # Convert to cm K for intensity formulae
u = sc.u          # Unified atomic mass unit (SI) = 1.66053904e-27 kg
pi = sc.pi        # pi = 3.141592653589793
Ryd = sc.Rydberg  # Rydberg constant