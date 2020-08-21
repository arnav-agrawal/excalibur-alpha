#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:11:36 2020

@author: arnav

File that the user of our package would use
"""

# Parameters
molecule = 'C2H4'
database = 'HITRAN'

import Download_Line_List
import Cthulhu_Refactored
import plot
import numpy as np
import re
import os

P = 1    # Bar
T = 2000   # K

# Download line list
Download_Line_List.summon(database=database, molecule=molecule)

# Create cross section
#nu, sigma = Cthulhu_Refactored.create_cross_section(input_dir = '../input/', database = database, molecule = molecule, log_pressure = np.log10(P), temperature = T, pressure_broadening='Burrows')

# Plot cross section
#plot.plot_results(nu_arr = nu, sigma_arr = sigma, molecule = molecule, temperature = T, log_pressure = np.log10(P))
    