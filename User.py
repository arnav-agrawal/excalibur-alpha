#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:11:36 2020

@author: arnav

File that the user of our package would use
"""

# Parameters for cross_section() function
input_directory = '/Volumes/Seagate Backup/input/H2O  |  (1H2-16O)/BT2/'   # Folder containing all downloaded data... Ex: 


import Download_Line_List
import Cthulhu_Refactored

Download_Line_List.summon(user_friendly = False, data_base = 'exomol', molecule = 'H2O', linelist='BT2')   # Download H2O2 from hitemp

#Cthulhu_Refactored.create_cross_section(input_directory, log_pressure = 0, temperature = 1000, pressure_broadening='air')