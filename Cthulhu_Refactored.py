#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:16:59 2020

@author: arnav
"""

import os
import numpy as np
import pandas as pd
import re
import scipy.constants as sc


from Voigt import Voigt_width, Generate_Voigt_grid_molecules, gamma_L_VALD, gamma_L_impact, analytic_alkali
from calculations import find_index, prior_index, bin_cross_section_atom, bin_cross_section_molecule
from calculations import produce_total_cross_section_EXOMOL, produce_total_cross_section_HITRAN
from calculations import produce_total_cross_section_VALD_atom, produce_total_cross_section_VALD_molecule

"""
def def_preferences():
        
    condor_run = False
    
    #***** Frequency grid *****#
    nu_min = 1.0                     # Computational grid min wavenumber (cm^-1)
    nu_max = 30000.0                 # Computational grid max wavenumber (cm^-1)
    nu_out_min = 200.0               # Output cross section grid min wavenumber (cm^-1)
    nu_out_max = 25000.0             # Output cross section grid max wavenumber (cm^-1)
    nu_ref = [1.0e2, 1.0e3, 1.0e4]   # Wavenumbers for reference Voigt widths
    dnu_out = 0.01                   # Linear spacing of output cross section (cm^-1)

    #***** Global intensity cut-off *****#
    S_cut = 1.0e-100   # Discard transitions with S < S_cut

    # Pressure broadening settings *****#

    X_H2 = 0.85           # Mixing ratio of H2 (solar)
    X_He = 0.15           # Mixing ratio of He (solar)
    gamma_0 = 0.07        # If fixed broadening chosen, use this Lorentzian HWHM
    n_L = 0.50            # If fixed broadening chosen, use this temperature exponent

    T_ref = 296.0   # Reference temperature for broadening parameters
    P_ref = 1.0     # Reference temperature for EXOMOL broadening parameters (bar) - HITRAN conversion from atm already pre-handled

    #***** Voigt profile computation settings *****#
    Voigt_sub_spacing = (1.0/6.0)  # Spacing of fine grid as a multiple of reference Voigt width
    Voigt_cutoff = 500.0           # Number of Voigt widths at which to stop computation
    N_alpha_samples = 500          # Number of values of alpha for precomputation of Voigt functions

    if (calculation_type == 'molecule'): cut_max = 30.0  # Absolute maximum wavenumber cutoff (cm^-1)
    if (calculation_type == 'atom'): cut_max = 1000.0    # Absolute maximum wavenumber cutoff (cm^-1)
    

def def_constants():

    #***** Define physical constants *****#

    c = sc.c     # Speed of light (SI) = 299792458.0 m s^-1
    kb = sc.k    # Boltzmann constant (SI) = 1.38064852e-23 m^2 kg s^-2 K^-1
    h = sc.h     # Planck's constant (SI) = 6.62607004e-34 m^2 kg s^-1
    m_e = sc.m_e # Electron mass (SIT) = 9.10938356e-31 kg
    c2 = h*c/kb  # Second radiation constant (SI) = 0.0143877736 m K  
    c2 *= 100.0  # Put in cm K for intensity formula
    u = sc.u     # Unified atomic mass unit (SI) = 1.66053904e-27 kg
    pi = sc.pi   # pi = 3.141592653589793
"""    
    
    
def check_molecule(molecule):
    match = re.match('^[A-Z]{1}[a-z]?$', molecule)
    
    if match: return True
    else: return False


def mass():
    return
    
    
def load_ExoMol(input_directory):
    print("Loading ExoMol format")
    
    # Read in states file (EXOMOL only)
    states_file_name = [filename for filename in os.listdir(input_directory) if filename.endswith('.states')]
    states_file = pd.read_csv(input_directory + states_file_name[0], sep = '\s+', header=None)
    E = np.array(states_file[1])
    g = np.array(states_file[2])
    J = np.array(states_file[3]).astype(np.int64)
    
    del states_file  # Delete file to free up memory    
    
    return E, g, J

def load_VALD():
    # Needs work
    nu_0, gf, E_low, E_up, J_low, l_low, l_up, gamma_nat, gamma_vdw = (np.array([]) for _ in range(9))
    
    for i in range(len(linelist_files)):
        
        # Read in VALD transitions files (also contains broadening parameters)
        trans_file = pd.read_csv(input_directory + linelist_files[i], sep = ' ', header=None, skiprows=1,
                                 dtype={'nu_0': np.float64, 'gf': np.float64, 
                                        'E_low': np.float64, 'E_up': np.float64,
                                        'J_low': np.float64, 'J_up': np.float64,
                                        'l_low': np.int64, 'l_up': np.int64, 
                                        'gamma_nat': np.float64, 
                                        'gamma_vdw': np.float64})
    
        # Merge linelists
        nu_0 = np.append(nu_0, np.array(trans_file[0]))
        gf = np.append(gf, np.array(trans_file[1]))
        E_low = np.append(E_low, np.array(trans_file[2]))
        E_up = np.append(E_up, np.array(trans_file[3]))
        J_low = np.append(J_low, np.array(trans_file[4])).astype(np.int64)
        l_low = np.append(l_low, np.array(trans_file[6]))
        l_up = np.append(l_up, np.array(trans_file[7]))
        gamma_nat = np.append(gamma_nat, np.array(trans_file[8]))
        gamma_vdw = np.append(gamma_vdw, np.array(trans_file[9]))
        
        # If transitions are not in increasing wavenumber order, rearrange
        order = np.argsort(nu_0)  # Indices of nu_0 in increasing order
        nu_0 = nu_0[order]
        gf = gf[order]
        E_low = E_low[order]
        E_up = E_up[order]
        J_low = J_low[order]
        l_low = l_low[order]
        l_up = l_up[order]
        gamma_nat = gamma_nat[order]
        gamma_vdw = gamma_vdw[order]
        
        del trans_file  # Delete file to free up memory
        
    return


def load_pf(input_directory):
    print("Loading partition functions")
    pf_file_name = [filename for filename in os.listdir(input_directory) if filename.endswith('.pf')]
    pf_file = pd.read_csv(input_directory + pf_file_name[0], sep= ' ', header=None, skiprows=1)
    T_pf_raw = np.array(pf_file[0]).astype(np.float64)
    Q_raw = np.array(pf_file[1])

    del pf_file   # Delete file to free up memory
    
    return T_pf_raw, Q_raw


def det_broad(input_directory):
    if 'H2.broad' in os.listdir(input_directory) and 'He.broad' in os.listdir(input_directory):
        broadening = 'H2-He'
        
    elif 'air.broad' in os.listdir(input_directory):
        broadening = 'air'
        
    else:
        broadening = 'Burrows'
        # To do: Create a Burrows broadening file and add it to directory
        
    return broadening


def read_H2_He(input_directory):
    # Read in H2 broadening file
    broad_file_H2 = pd.read_csv(input_directory + 'H2.broad', sep = ' ', header=None, skiprows=1)
    J_max_H2 = int(np.max(np.array(broad_file_H2[0])))
    gamma_0_H2 = np.array(broad_file_H2[1])
    n_L_H2 = np.array(broad_file_H2[2])
    
    # Read in He broadening file
    broad_file_He = pd.read_csv(input_directory + 'He.broad', sep = ' ', header=None, skiprows=1)
    J_max_He = int(np.max(np.array(broad_file_He[0])))
    gamma_0_He = np.array(broad_file_He[1])
    n_L_He = np.array(broad_file_He[2])

    # Take maximum J'' value for which broadening is a function of J to be lowest for which complete data available
    J_max = np.max(np.array([J_max_H2, J_max_He]))

    # If broadening files not of same length, extend shortest file to same length as longest
    if (J_max_H2 < J_max):
            
        for i in range (J_max_H2, J_max):
                
            gamma_0_H2 = np.append(gamma_0_H2, gamma_0_H2[-1])    # Extended values equal to final value 
            n_L_H2 = np.append(n_L_H2, n_L_H2[-1])                # Extended values equal to final value 
            
    if (J_max_He < J_max):
        
        for i in range (J_max_He, J_max):
                
            gamma_0_He = np.append(gamma_0_He, gamma_0_He[-1])    # Extended values equal to final value 
            n_L_He = np.append(n_L_He, n_L_He[-1])                # Extended values equal to final value 
                
    del broad_file_H2, broad_file_He   # Delete files to free up memory
    
    return J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He
    
    
def read_air(input_directory):
    # Read in air broadening file
    broad_file_air = pd.read_csv(input_directory + 'air.broad', sep = ' ', header=None, skiprows=1)
    J_max = int(np.max(np.array(broad_file_air[0])))
    gamma_0_air = np.array(broad_file_air[1])
    n_L_air = np.array(broad_file_air[2])
    
    del broad_file_air   # Delete file to free up memory  
    
    return J_max, gamma_0_air, n_L_air
    
def read_Burrows(input_directory):
    
    # Read in air broadening file
    broad_file_Burrows = pd.read_csv(input_directory + 'Burrows.broad', sep = ' ', header=None, skiprows=1)
    J_max = int(np.max(np.array(broad_file_Burrows[0])))
    gamma_0_Burrows = np.array(broad_file_Burrows[1])
    #n_L_Burrows = np.array(broad_file_Burrows[2])       # Not really needed, as temperature exponent = 0 for all J''
    
    del broad_file_Burrows   # Delete file to free up memory
    
    return J_max, gamma_0_Burrows
    

def compute_pressure_broadening():
    return

def create_wavelength_grid():
    return

def compute_Voigt():
    return
    
def plot_results():
    return

    
def create_cross_section(input_dir, database, molecule, isotope, linelist, temperature, pressure):
    # Will need if-else for condor_run = True or False
    
    is_molecule = check_molecule(molecule)
    
    T_pf_raw, Q_raw = load_pf(input_dir)
    
    
    if database == 'exomol':
        E, g, J = load_ExoMol(input_dir)
    
    if database == 'hitran':
        return
    
    if database == 'hitemp':
        return
    
    if database == 'vald':
        return
    
    return
