#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:11:36 2020

@author: arnav

File that the user of our package would use
"""

# Parameters for cross_section() function
input_directory = '/Volumes/Seagate Backup/input/HCN  ~  (1H-12C-14N)/Harris/'   # Folder containing all downloaded data... Ex: 


import Download_Line_List
import Cthulhu_Refactored
import re
#import plot

Download_Line_List.summon()

#Cthulhu_Refactored.create_cross_section(input_dir = '/Volumes/Seagate Backup/input/', database = 'exomol', molecule = 'HCN', log_pressure = 0, temperature = 1000)

#plot.plot_results(wl, sigma)
