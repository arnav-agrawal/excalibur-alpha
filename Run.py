#***** Example script to run EXCALIBUR *****#

import numpy as np

from excalibur.core import summon
from excalibur.core import compute_cross_section
from excalibur.plot import plot_sigma_wl

# Parameters
molecule = 'H2O'
database = 'HITEMP'

P = 1       # Pressure (bar)
T = 1200    # Temperature (K)

# Download line list
summon(database=database, molecule=molecule)

# Create cross section
nu, sigma = compute_cross_section(input_dir = './input/', database = database, 
                                  molecule = molecule, log_pressure = np.log10(P), 
                                  temperature = T)

# Plot cross section
plot_sigma_wl(nu_arr = nu, sigma_arr = sigma, molecule = molecule, temperature = T, 
              log_pressure = np.log10(P), database = database)
    