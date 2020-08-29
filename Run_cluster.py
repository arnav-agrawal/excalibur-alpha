#***** Example script to batch-run EXCALIBUR on a cluster *****#

import numpy as np

from excalibur.core import summon
from excalibur.core import compute_cross_section

# Parameters
molecule = 'H2O'
database = 'HITRAN'

P = [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e2]    # Pressure (bar)
T = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0,                 # Temperature (K)
     900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0,
     2500.0, 3000.0, 3500.0]

# Download line list (Only need to run this once to download the line list!)
# summon(database=database, molecule=molecule)

# Create cross section
nu, sigma = compute_cross_section(input_dir = './input/', database = database, 
                                  molecule = molecule, log_pressure = np.log10(P), 
                                  temperature = T, cluster_run = True)

