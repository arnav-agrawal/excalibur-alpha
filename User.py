#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:11:36 2020

@author: arnav

File that the user of our package would use
"""

# Parameters
molecule = 'TiO'
database = 'exomol'
ionization = 1

import Download_Line_List
import Cthulhu_Refactored
import plot
import numpy as np
import re
import os

P = 1    # Bar
T = 2000   # K

# Download line list
#Download_Line_List.summon(database=database, molecule=molecule)

# Create cross section

#nu, sigma = Cthulhu_Refactored.create_cross_section(input_dir = '../input/', database = database, 
                                                    #molecule = molecule, ionization_state = ionization, 
                                                    #log_pressure = np.log10(P), temperature = T)

# Plot cross section
plot.plot_results(file = '../TiO initial/TiO_T2000K_log_P0.0_sigma.txt', molecule = molecule, temperature = T, log_pressure = np.log10(P))
    