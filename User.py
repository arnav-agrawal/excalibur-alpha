#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:11:36 2020

@author: arnav

File that the user of our package would use
"""

# Parameters for cross_section() function
input_directory = '/Volumes/Seagate Backup/input/HNO3  |  (1H-14N-16O3)/AIJS'   # Folder containing all downloaded data... Ex:
output_directory = ''
cluster_run = False
log_pressure = 0.0
temperature = 1000.0
nu_out_min = 1
nu_out_max = 30,000
dnu_out = 0.01
pressure_broadening = 'default'
X_H2 = 0.85
X_He = 0.15
Voigt_cutoff = (1.0/6.0)
Voigt_sub_spacing = 500
N_alpha_samples = 500
S_cut = 1.0e-100
cut_max = 30.0 # special case for atoms is 1000 which will be dealt with later


import Download_Line_List

Download_Line_List.summon(False, 'exomol', 'HCN')   # Download NO from hitemp

#cross_section(input_directory, log_pressure, temperature, output_directory, cluster_run, nu_min, 
#              nu_max, dnu, pressure_broadening, Voigt_cutoff, Voigt_sub_spacing, N_alpha_samples, S_cut)