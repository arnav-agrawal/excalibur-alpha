#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 22:38:44 2020

@author: arnav
"""

import scipy.constants as sc

#***** Frequency grid *****#

#nu_min = 1.0                     # Computational grid min wavenumber (cm^-1)
#nu_max = 30000.0                 # Computational grid max wavenumber (cm^-1)
Nu_ref = [1.0e2, 1.0e3, 1.0e4]   # Wavenumbers for reference Voigt widths

# Pressure broadening settings *****#

gamma_0 = 0.07        # If fixed broadening chosen, use this Lorentzian HWHM
n_L = 0.50            # If fixed broadening chosen, use this temperature exponent

T_ref = 296.0   # Reference temperature for broadening parameters
P_ref = 1.0     # Reference temperature for EXOMOL broadening parameters (bar) - HITRAN conversion from atm already pre-handled

#***** Define physical constants *****#

c = sc.c     # Speed of light (SI) = 299792458.0 m s^-1
kb = sc.k    # Boltzmann constant (SI) = 1.38064852e-23 m^2 kg s^-2 K^-1
h = sc.h     # Planck's constant (SI) = 6.62607004e-34 m^2 kg s^-1
m_e = sc.m_e # Electron mass (SIT) = 9.10938356e-31 kg
c2 = h*c/kb  # Second radiation constant (SI) = 0.0143877736 m K  
c2 *= 100.0  # Put in cm K for intensity formula
u = sc.u     # Unified atomic mass unit (SI) = 1.66053904e-27 kg
pi = sc.pi   # pi = 3.141592653589793