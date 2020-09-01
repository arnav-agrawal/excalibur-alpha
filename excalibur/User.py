#***** Example script to run EXCALIBUR *****#

# Parameters
molecule = 'H2O'
database = 'HITRAN'
ionization = 1

from excalibur import Download_Line_List
from excalibur import Cthulhu_Refactored
from excalibur import plot
import numpy as np
import re
import os

P = 1    # Bar
T = 1200 
#T_1 = 1200   # K
#T_2 = 2000   # K

# Download line list
Download_Line_List.summon(database=database, molecule=molecule, ionization_state = ionization)

# Create cross section
#nu, sigma = Cthulhu_Refactored.create_cross_section(input_dir = '../input/', database = database, 
                                                    #molecule = molecule, ionization_state = ionization, 
                                                    #log_pressure = np.log10(P), temperature = T)

# Plot cross section
#plot.plot_results(nu_arr = nu, sigma_arr = sigma, molecule = molecule, temperature = T, log_pressure = np.log10(P))
    
#nu_1, sigma_1 = Cthulhu_Refactored.create_cross_section(input_dir = '../input/', database = database, 
                                                    #molecule = molecule, ionization_state = ionization, log_pressure = np.log10(P), temperature = T_1)

#nu_2, sigma_2 = Cthulhu_Refactored.create_cross_section(input_dir = '../input/', database = database, 
                                                    #molecule = molecule, ionization_state = ionization, log_pressure = np.log10(P), temperature = T_2)


#plot.compare_cross_sections(molecule = molecule, label_1 = '1200 K', label_2 = '2000 K', 
                            #nu_arr_1 = nu_1, nu_arr_2 = nu_2, sigma_arr_1 = sigma_1, sigma_arr_2 = sigma_2)
