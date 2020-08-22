#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:11:36 2020

@author: arnav

File that the user of our package would use
"""

# Parameters
molecule = 'Na'
database = 'vald'
ionization = 1

import Download_Line_List
import Cthulhu_Refactored
import plot
import numpy as np
import re
import os

P = 1    # Bar
T_1 = 1200   # K
T_2 = 2000   # K

# Download line list
Download_Line_List.summon(database=database, molecule=molecule)

# Create cross section
nu_1, sigma_1 = Cthulhu_Refactored.create_cross_section(input_dir = '../input/', database = database, 
                                                    molecule = molecule, ionization_state = ionization, log_pressure = np.log10(P), temperature = T_1)

nu_2, sigma_2 = Cthulhu_Refactored.create_cross_section(input_dir = '../input/', database = database, 
                                                    molecule = molecule, ionization_state = ionization, log_pressure = np.log10(P), temperature = T_2)


# Plot cross section
#plot.plot_results(nu_arr = nu, sigma_arr = sigma, molecule = molecule, temperature = T, log_pressure = np.log10(P))
    
plot.compare_cross_sections(molecule = molecule, label_1 = '1200 K', label_2 = '2000 K', 
                            nu_arr_1 = nu_1, nu_arr_2 = nu_2, sigma_arr_1 = sigma_1, sigma_arr_2 = sigma_2)