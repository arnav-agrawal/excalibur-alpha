#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:11:36 2020

@author: arnav

File that the user of our package would use
"""

# Parameters for cross_section() function


import Download_Line_List
import Cthulhu_Refactored
import plot

Download_Line_List.summon(molecule = 'CO2', database='exomol')

#nu, sigma = Cthulhu_Refactored.create_cross_section(input_dir = '/Volumes/Seagate Backup/input/', database = 'exomol', molecule = 'HCN', log_pressure = [0], temperature = [1000])

#plot.plot_results(nu_arr = nu, sigma_arr = sigma, molecule = 'VO', temperature = 3000, log_pressure = 2)
