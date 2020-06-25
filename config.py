# Contains configuraton settings used in Cthulhu.py and subsidiary modules

import os
import numpy as np
import scipy.constants as sc

# Set cross section computation for molecule or atomic species

calculation_type = 'molecule'    # Options: molecule / atom

#***** Specify species information *****#

species = 'H2O'
species_id = '1H2-16O'
linelist = 'HITRAN'
file_format = 'HITRAN'        # Options: EXOMOL / HITRAN / VALD

#***** Input and output directories *****#


if (calculation_type == 'molecule'):
    input_directory = '../input/' + species + '-H2(16O)' + '/' + linelist + '/'    # Folder location containing linelist etc
    output_directory = '../output/' # Folder location containing output cross sections

if (calculation_type == 'atom'):
    input_directory = '../../Atoms/' + species + '/input/' + linelist + '/'    # Folder location containing linelist etc
    output_directory = '../../Atoms/' + species + '/output/' + linelist + '/'  # Folder location containing output cross sections


#***** Linelist input files *****#

linelist_files = [filename for filename in os.listdir(input_directory) if filename.endswith('.trans')]

#***** Program settings *****#
compute_Voigt = True
make_cross_section = True
write_output = True
plot_results = True
condor_run = False

#***** Conditions for cross section calculation *****#

# For a single point in P-T space:
if (condor_run == False):
    log_P_arr = np.array([0.0])        # log_10 (Pressure/bar)
    T_arr = np.array([1000.0])         # Temperature (K)

# For multiple (P,T) points on a Condor run
elif (condor_run == True):
    log_P_arr = np.array([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])   # log_10 (Pressure/bar)
    T_arr = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0,   # Temperature (K)
                      900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 
                      2500.0, 3000.0, 3500.0])

#***** Frequency grid *****#

nu_min = 1.0                     # Computational grid min wavenumber (cm^-1)
nu_max = 30000.0                 # Computational grid max wavenumber (cm^-1)
nu_out_min = 200.0               # Output cross section grid min wavenumber (cm^-1)
nu_out_max = 25000.0             # Output cross section grid max wavenumber (cm^-1)
nu_ref = [1.0e2, 1.0e3, 1.0e4]   # Wavenumbers for reference Voigt widths
dnu_out = 0.01                   # Linear spacing of output cross section (cm^-1)

#***** Global intensity cut-off *****#
S_cut = 1.0e-100   # Discard transitions with S < S_cut

# Pressure broadening settings *****#

broadening = 'air'  # Options: H2-He / air / fixed
X_H2 = 0.85           # Mixing ratio of H2 (solar)
X_He = 0.15           # Mixing ratio of He (solar)
gamma_0 = 0.07        # If fixed broadening chosen, use this Lorentzian HWHM
n_L = 0.50            # If fixed broadening chosen, use this temperature exponent

T_ref = 296.0   # Reference temperature for broadening parameters
P_ref = 1.0     # Reference temperature for EXOMOL broadening parameters (bar) - HITRAN conversion from atm already pre-handled

#***** Voigt profile computation settings *****#
Voigt_sub_spacing = (1.0/6.0)  # Spacing of fine grid as a multiple of reference Voigt width
Voigt_cutoff = 500.0           # Number of Voigt widths at which to stop computation
N_alpha_samples = 500          # Number of values of alpha for precomputation of Voigt functions

if (calculation_type == 'molecule'): cut_max = 30.0  # Absolute maximum wavenumber cutoff (cm^-1)
if (calculation_type == 'atom'): cut_max = 1000.0    # Absolute maximum wavenumber cutoff (cm^-1)

#***** Define physical constants *****#

c = sc.c     # Speed of light (SI) = 299792458.0 m s^-1
kb = sc.k    # Boltzmann constant (SI) = 1.38064852e-23 m^2 kg s^-2 K^-1
h = sc.h     # Planck's constant (SI) = 6.62607004e-34 m^2 kg s^-1
m_e = sc.m_e # Electron mass (SIT) = 9.10938356e-31 kg
c2 = h*c/kb  # Second radiation constant (SI) = 0.0143877736 m K  
c2 *= 100.0  # Put in cm K for intensity formula
u = sc.u     # Unified atomic mass unit (SI) = 1.66053904e-27 kg
pi = sc.pi   # pi = 3.141592653589793

#***** Identify mass of chosen molecule *****#

masses = {'1H2-16O':       18.010565, '12C-1H4':     16.031300, '14N-1H3':   17.026549, '1H-12C-14N':   27.010899,
          '12C-16O':       27.994915, '12C-16O2':    43.989830, '32S-16O2':  63.961901, '32S-16O3':     79.956820,
          '31P-1H3':       33.997238, '16O2':        31.989830, '16O3':      47.984745, '14N2':         28.006148,
          '48Ti-16O':      63.942861, '51V-16O':     66.938871, '27Al-16O':  42.976454, '28Si-16O':     43.971842,
          '40Ca-16O':      55.957506, '48Ti-1H':     48.955771, '56Fe-1H':   56.942762, '7Li-1H':       8.0238300,
          '45Sc-1H':       45.963737, '24Mg-1H':     24.992867, '23Na-1H':   23.997594, '27Al-1H':      27.989480,
          '52Cr-1H':       52.948333, '40Ca-1H':     40.970416, '9Be-1H':    10.020007, '28Si-1H':      28.984752,
          '14N-1H':        15.010899, '12C-1H':      13.007825, '16O-1H':    17.002740, '32S-1H':       32.979896,
          '23Na-35Cl':     57.958622, '39K-35Cl':    73.932560, '1H-35Cl':   35.976678, '1H3_p':        3.0234750,
          '14N-16O':       29.997989, '14N-32S':     45.975145, '31P-16O':   46.968676, '31P-32S':      62.945833,
          '12C-31P':       42.973762, '12C-32S':     43.972071, '12C-14N':   26.003074, '31P-14N':      44.976836,
          '1H-14N-16O3':   62.995644, '12C2-1H2':    26.015650, '12C2-1H4':  28.031300, '1H2-32S':      33.987721,
          '1H2-12C-16O':   30.010565, '12C2':        21.000000, '14N2-16O':  44.001062, '1H2-16O2':     34.005479,
          '12C-1H3-19F':   34.021878, '28Si-1H4':    32.008227, '14N-16O2':  45.992904, '1H-19F':       20.006229,
          '1H-79Br':       79.926160, '1H-127I':     127.91230, '35Cl-16O':  50.963768, '12C2-1H6':     30.046950,
          '16O-12C-32S':   59.966986, '32S-19F6':    145.96249, '1H-16O2':   32.997655, '1H-16O-35Cl':  51.971593,
          '12C-1H3-35Cl':  49.992328, '14N-16O_p':   29.997989, '12C-19F4':  87.993616, '12C-1H4-16O':  32.026215,  
          '12C-16O-19F2':  65.991722, '1H-16O-79Br': 95.921076, '12C4-1H2':  50.015650, '12C-1H3-79Br': 93.941811,
          '1H2-12C-16O2':  46.005480, '1H-12C3-14N': 51.010899, '12C2-14N2': 52.006148, '12C2-1H3-14N': 41.026549,
          '35Cl-16O3-14N': 96.956672, '16O':         15.994915, '7Li':       7.0160040, '23Na':         22.989770,  
          '12C-16O-35Cl2': 97.932620, '39K':         38.963707, '85Rb':      84.911789, '133Cs':        132.90545,
          '32S-1H':        32.979896, '12C-1H':      13.007825, '14N-1H':    15.010899, '28Si-1H':      28.984752,
          '56Fe':          55.934942, '48Ti':        47.947947, '24Mg':      23.985042, '1H':           1.0078250,
          '28Si':          27.976927
              }



