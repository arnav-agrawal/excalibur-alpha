#***** Example script to run EXCALIBUR *****#

import numpy as np

from excalibur.core import summon
from excalibur.core import compute_cross_section
from excalibur.plot import plot_sigma_wl

# Parameters
species = 'Na'
isotope = 'default'
ion = 1
database = 'VALD'

P = 1e-3       # Pressure (bar)
T = 2000       # Temperature (K)

# Download line list
summon(species = species, database = database, ionization_state=ion, 
       isotope = isotope, VALD_data_dir = './VALD Line Lists/')

# Create cross section
nu, sigma = compute_cross_section(input_dir = './input/', database = database, 
                                  species = species, log_pressure = np.log10(P), 
                                  temperature = T, ionization_state = ion, isotope = isotope,
                                  N_cores = 1, nu_out_min = 200, nu_out_max = 50000, dnu_out = 0.01)

# Plot cross section
plot_sigma_wl(nu_arr = nu, sigma_arr = sigma, species = species, ionization = ion, temperature = T, 
              log_pressure = np.log10(P), database = database, plot_dir = './plots/')
