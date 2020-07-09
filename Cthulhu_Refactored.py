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

from hapi import molecularMass


from Voigt import Voigt_width, Generate_Voigt_grid_molecules, gamma_L_VALD, gamma_L_impact, analytic_alkali
from calculations import find_index, prior_index, bin_cross_section_atom, bin_cross_section_molecule
from calculations import produce_total_cross_section_EXOMOL, produce_total_cross_section_HITRAN
from calculations import produce_total_cross_section_VALD_atom, produce_total_cross_section_VALD_molecule

from constants import nu_min, nu_max, nu_out_min, nu_out_max, nu_ref, dnu_out
from constants import u, c, kb, log_P_arr, T_arr, cut_max
from constants import species_id, masses, T_ref, P_ref, X_H2, X_He
from constants import Voigt_sub_spacing, N_alpha_samples, Voigt_cutoff
    
    
def check_molecule(molecule):
    match = re.match('^[A-Z]{1}[a-z]?$', molecule)
    
    if match: return True
    else: return False


def mass():
    # Will need to find the .def file from ExoMol if the line list is not HITRAN or HITEMP
    # otherwise can just use the molecularMass() function in hapi
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
    
    # Read in Burrows broadening file
    broad_file_Burrows = pd.read_csv(input_directory + 'Burrows.broad', sep = ' ', header=None, skiprows=1)
    J_max = int(np.max(np.array(broad_file_Burrows[0])))
    gamma_0_Burrows = np.array(broad_file_Burrows[1])
    #n_L_Burrows = np.array(broad_file_Burrows[2])       # Not really needed, as temperature exponent = 0 for all J''
    
    del broad_file_Burrows   # Delete file to free up memory
    
    return J_max, gamma_0_Burrows


def interpolate_pf():
    
    #***** Interpolate (and extrapolate) partition function to a fine grid *****#
    pf_spline = Interp(T_pf_raw, Q_raw, k=5)
    T_pf_fine = np.linspace(1.0, 10000.0, 9999)       # New temperature grid (extrapolated to 4000K)
    Q_fine = pf_spline(T_pf_fine)                    # Extrapolated partition function
    i_T = np.argmin(np.abs(T_pf_fine - T))           # Index of partition function temperature array closest to desired temperature
    i_T_ref = np.argmin(np.abs(T_pf_fine - T_ref))   # Index of partition function temperature array closest to reference temperature
    Q_T = Q_fine[i_T]                                # Partition function at given temperature, Q(T)
    Q_T_ref = Q_fine[i_T_ref]                        # Partition function at reference temperature, Q(T_ref)
    
    return Q_T, Q_T_ref

def compute_pressure_broadening_atom():
    
    if (species in ['Li', 'Na','K', 'Rb', 'Cs']):  # Special treatments for alkali van der waals widths
                
        gamma_0_H2 = np.zeros(len(nu_0))
        gamma_0_He = np.zeros(len(nu_0))
        n_L_H2 = np.zeros(len(nu_0))
        n_L_He = np.zeros(len(nu_0))
                
        for i in range(len(nu_0)):
                    
            if (gamma_vdw[i] != 0.0):  # For transitions with a VALD broadening value 
                        
                gamma_0_H2[i], n_L_H2[i] = gamma_L_VALD(gamma_vdw[i], (m/u), 'H2')
                gamma_0_He[i], n_L_He[i] = gamma_L_VALD(gamma_vdw[i], (m/u), 'He')
                    
            elif (gamma_vdw[i] == 0.0):  # For transitions without a VALD broadening value 
                        
                gamma_0_H2[i], n_L_H2[i] = gamma_L_impact(E_low[i], E_up[i], l_low[i], l_up[i], species, (m/u), 'H2')
                gamma_0_He[i], n_L_He[i] = gamma_L_impact(E_low[i], E_up[i], l_low[i], l_up[i], species, (m/u), 'He')
                        
                # Remove transitions where Hydrogenic approximation breaks down
          #      accept_condition = np.where(gamma_0_H2 != -1.0)  # Transitions without a VALD broadening value
                        
          #      nu_0 = nu_0[accept_condition]
          #      gf = gf[accept_condition]
          #      E_low = E_low[accept_condition]
          #      E_up = E_up[accept_condition]
          #      J_low = J_low[accept_condition]
          #      l_low = l_low[accept_condition]
          #      l_up = l_up[accept_condition]
          #      n_L_H2 = n_L_H2[accept_condition]
          #      n_L_He = n_L_He[accept_condition]
          #      gamma_0_He = gamma_0_He[accept_condition]
          #      gamma_0_H2 = gamma_0_H2[accept_condition]
                
    else:  # For non-alkali species
                
          #      accept_condition = np.where(log_gamma_vdw != 0.0)  # Transitions without a VALD broadening value 
                
          #      log_gamma_vdw = log_gamma_vdw[accept_condition]
          #      nu_0 = nu_0[accept_condition]
          #      gf = gf[accept_condition]
          #      E_low = E_low[accept_condition]
          #      E_up = E_up[accept_condition]
          #      J_low = J_low[accept_condition]
          #      l_low = l_low[accept_condition]
          #      l_up = l_up[accept_condition]
                
        gamma_0_H2, n_L_H2 = gamma_L_VALD(gamma_vdw, (m/u), 'H2')
        gamma_0_He, n_L_He = gamma_L_VALD(gamma_vdw, (m/u), 'He')


def compute_H2_He_broadening():
    gamma = (gamma_0_H2 * np.power((T_ref/T), n_L_H2) * (P/P_ref) * X_H2 +   # H2+He Lorentzian HWHM for given T, P, and J (ang. mom.)
             gamma_0_He * np.power((T_ref/T), n_L_He) * (P/P_ref) * X_He)    # Note that these are only a function of J''
    
    return gamma
    
def compute_air_broadening():
    gamma = (gamma_0_air * np.power((T_ref/T), n_L_air) * (P/P_ref))      # Air-broadened Lorentzian HWHM for given T, P, and J (ang. mom.)

    return gamma
    
def compute_Burrows_broadening():
    gamma = (gamma_0_Burrows * (P/P_ref))      # Equation (15) in Sharp & Burrows (2007)  
    
    return gamma

def create_wavelength_grid_atom():
    # First, we need to find values of gamma_V for reference wavenumber (1000 cm^-1)
    alpha_ref = np.sqrt(2.0*kb*T*np.log(2)/m) * (np.array(nu_ref[2])/c) # Doppler HWHM at reference wavenumber
    gamma_ref = np.min(gamma)                                           # Find minimum value of Lorentzian HWHM
    gamma_V_ref = Voigt_width(gamma_ref, alpha_ref)                     # Reference Voigt width
    
    # Calculate Voigt width for each transition
    alpha = np.sqrt(2.0*kb*T*np.log(2)/m) * (np.array(nu_0)/c)   # Doppler HWHM for each transition
    gamma_V = Voigt_width(gamma, alpha)
    
    # Now compute properties of computational (fine) and output (coarse) wavenumber grid
    
    # Wavenumber spacing of of computational grid (smallest of gamma_V_ref/6 or 0.01cm^-1)
    dnu_fine = np.minimum(gamma_V_ref*Voigt_sub_spacing, dnu_out)      
    
    # Number of points on fine grid (rounded)
    N_points_fine = int((nu_max-nu_min)/dnu_fine + 1)
    
    # Adjust dnu_fine slightly to match an exact integer number of grid spaces
    dnu_fine = (nu_max-nu_min)/(N_points_fine - 1)
    
    cutoffs = np.zeros(len(nu_0))   # Line wing cutoffs for each line
    
    # Line cutoffs at @ min(500 gamma_V, 1000cm^-1)
    for i in range(len(nu_0)):
        cutoffs[i] = dnu_fine * (int((Voigt_cutoff*gamma_V[i])/dnu_fine))

        if (cutoffs[i] >= cut_max): cutoffs[i] = cut_max
                
        # Special cases for alkali resonant lines
        if   ((species == 'Na') and (int(nu_0[i]) in [16978, 16960])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        elif ((species == 'K') and  (int(nu_0[i]) in [13046, 12988])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        #elif ((species == 'Li') and (int(nu_0[i]) in [14908, 14907])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        #elif ((species == 'Rb') and (int(nu_0[i]) in [12820, 12582])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        #elif ((species == 'Cs') and (int(nu_0[i]) in [11735, 9975])):  cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
            
        # Calculate detuning frequencies for Na and K resonance lines
        if (species == 'Na'): nu_detune = 30.0 * np.power((T/500.0), 0.6)
        elif (species == 'K'): nu_detune = 20.0 * np.power((T/500.0), 0.6)
        else: nu_detune = cut_max
        
        # Evaluate number of frequency points for each Voigt function up to cutoff (one tail)
        N_Voigt_points = ((cutoffs/dnu_fine).astype(np.int64)) + 1  
        
        # Define start and end points of fine grid
        #nu_fine = np.linspace(nu_min, nu_max, N_points_fine)
        nu_fine_start = nu_min
        nu_fine_end = nu_max
        
        # Initialise output grid
        N_points_out = int((nu_out_max-nu_out_min)/dnu_out + 1)     # Number of points on coarse grid (uniform)
        nu_out = np.linspace(nu_out_min, nu_out_max, N_points_out)  # Create coarse (output) grid
        
        # Initialise cross section arrays on each grid
        sigma_fine = np.zeros(N_points_fine)    # Computational (fine) grid
        sigma_out = np.zeros(N_points_out)      # Coarse (output) grid

def create_wavelength_grid_molecule():
    # First, we need to find values of gamma_V for reference wavenumbers (100, 1000, 10000 cm^-1)
    nu_ref = np.array(nu_ref)   # The reference wavenumbers need to be a numpy array
    alpha_ref = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_ref/c)    # Doppler HWHM at reference wavenumbers 
    gamma_ref = np.min(gamma)                                           # Find minimum value of Lorentzian HWHM
    gamma_V_ref = Voigt_width(gamma_ref, alpha_ref)                     # Reference Voigt widths
    
    # Now compute properties of computational (fine) and output (coarse) wavenumber grids
            
    # Wavenumber spacing of three regions of computational grid (smallest of gamma_V_ref/6 or 0.01cm^-1)
    dnu_fine = np.minimum(gamma_V_ref*Voigt_sub_spacing, dnu_out*np.ones(3))   
    
    # Number of points on fine grid (three regions, rounded)
    N_points_fine_1 = int((nu_ref[1]-nu_min)/dnu_fine[0])
    N_points_fine_2 = int((nu_ref[2]-nu_ref[1])/dnu_fine[1])
    N_points_fine_3 = int((nu_max-nu_ref[2])/dnu_fine[2] + 1)
    N_points_fine = N_points_fine_1 + N_points_fine_2 + N_points_fine_3
    
    # Adjust dnu_fine slightly to match an exact integer number of grid spaces
    dnu_fine[0] = (nu_ref[1]-nu_min)/N_points_fine_1
    dnu_fine[1] = (nu_ref[2]-nu_ref[1])/N_points_fine_2
    dnu_fine[2] = (nu_max-nu_ref[2])/(N_points_fine_3 - 1)
    cutoffs = np.zeros(len(dnu_fine))   # Line wing cutoffs in three regions
    
    # Line cutoffs at @ min(500 gamma_V, 30cm^-1)
    for i in range(len(dnu_fine)):
                
        # If grid spacing below maximum dnu (0.01), set to 3000 dnu (~ 500 gamma_V)
        if (dnu_fine[i] < dnu_out):
            cutoffs[i] = Voigt_cutoff*(1.0/Voigt_sub_spacing)*dnu_fine[i] 
            
        # If at maximum dnu (0.01), set to min(500 gamma_V, 30cm^-1)
        else:
            cutoffs[i] = dnu_fine[i] * int(Voigt_cutoff*gamma_V_ref[i]/dnu_fine[i])   
            if (cutoffs[i] >= cut_max): cutoffs[i] = cut_max
            
    # Define start and end points of each fine grid
    nu_fine_1_start = nu_min
    nu_fine_1_end = (nu_ref[1] - dnu_fine[0])
    nu_fine_2_start = nu_ref[1]
    nu_fine_2_end = (nu_ref[2] - dnu_fine[1])
    nu_fine_3_start = nu_ref[2]
    nu_fine_3_end = nu_max
    
    #nu_fine_1 = np.linspace(nu_min, nu_ref[1], N_points_fine_1, endpoint=False)
    #nu_fine_2 = np.linspace(nu_ref[1], nu_ref[2], N_points_fine_2, endpoint=False)
    #nu_fine_3 = np.linspace(nu_ref[2], nu_max, N_points_fine_3)
    #nu_fine = np.concatenate([nu_fine_1, nu_fine_2, nu_fine_3])
    
    # Initialise output grid
    N_points_out = int((nu_out_max-nu_out_min)/dnu_out + 1)     # Number of points on coarse grid (uniform)
    nu_out = np.linspace(nu_out_min, nu_out_max, N_points_out)  # Create coarse (output) grid
        
    # Initialise cross section arrays on each grid
    sigma_fine = np.zeros(N_points_fine)    # Computational (fine) grid
    sigma_out = np.zeros(N_points_out)      # Coarse (output) grid
    
    #***** Pre-compute Voigt function array for molecules *****#
    
    print('Pre-computing Voigt profiles...')
    
    t1 = time.perf_counter()
        
    # First, create an array of values of alpha to approximate true values of alpha (N=500 log-spaced => max error of 0.5%)
    nu_sampled = np.logspace(np.log10(nu_ref[0]), np.log10(nu_max), N_alpha_samples)
    alpha_sampled = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_sampled/c)
    
    # Evaluate number of frequency points for each Voigt function in each spectral region - up to cutoff @ min(500 gamma_V, 30cm^-1)
    N_Voigt_points = ((cutoffs/dnu_fine).astype(np.int64)) + 1  
            
    # Pre-compute and store Voigt functions and first derivatives wrt alpha 
    if (compute_Voigt == True):
        Voigt_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))    # For H2O: V(51,500,3001)
        dV_da_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))    # For H2O: V(51,500,3001)
        dV_dnu_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))   # For H2O: V(51,500,3001)
        Generate_Voigt_grid_molecules(Voigt_arr, dV_da_arr, dV_dnu_arr, gamma, alpha_sampled, alpha_ref, cutoffs, N_Voigt_points)
        
    t2 = time.perf_counter()
    total1 = t2-t1
            
    print('Voigt profiles computed in ' + str(total1) + ' s')        


def run_cross_section():
            
    print('Generating cross section for ' + species + ' at P = ' + str(P) + ' bar, T = ' + str(T) + ' K')
        
    if (file_format == 'EXOMOL'):
                
        produce_total_cross_section_EXOMOL(linelist_files, input_directory,sigma_fine,
                                           nu_sampled, nu_ref, m, T, Q_T, N_points_fine_1,
                                           N_points_fine_2, N_points_fine_3, dnu_fine,
                                           N_Voigt_points, cutoffs, g, E, J, J_max, 
                                           alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr)
            
    elif (file_format == 'HITRAN'):
            
        produce_total_cross_section_HITRAN(linelist_files, input_directory, sigma_fine,
                                           nu_sampled, nu_ref, m, T, Q_T, Q_T_ref,
                                           N_points_fine_1, N_points_fine_2, N_points_fine_3,
                                           dnu_fine, N_Voigt_points, cutoffs, J_max, 
                                           alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr)
                
    elif (file_format == 'VALD'):
        
        if (calculation_type == 'molecule'):
                    
            produce_total_cross_section_VALD_molecule(sigma_fine, nu_sampled, nu_ref, nu_0, E_low, J_low,
                                                      gf, m, T, Q_T, N_points_fine_1, N_points_fine_2,
                                                      N_points_fine_3, dnu_fine, N_Voigt_points, cutoffs, 
                                                      J_max, alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr)
                
        elif (calculation_type == 'atom'):
                    
            produce_total_cross_section_VALD_atom(sigma_fine, nu_0, nu_detune, E_low, gf, m, T, Q_T,
                                                  N_points_fine, N_Voigt_points, alpha, gamma, cutoffs)
                    


        
    # Now bin cross section to output grid    
    print('Binning cross section to output grid...')
        
    if (calculation_type == 'atom'):
                
        bin_cross_section_atom(sigma_fine, sigma_out, nu_fine_start, 
                               nu_fine_end, nu_out, N_points_fine, N_points_out, 0)
            
    if (calculation_type == 'molecule'):
                
        bin_cross_section_molecule(sigma_fine, sigma_out, N_points_fine_1, N_points_fine_2,
                                   N_points_fine_3, nu_ref, nu_fine_1_start, nu_fine_1_end,
                                   nu_fine_2_start, nu_fine_2_end, nu_fine_3_start, nu_fine_3_end,
                                   nu_out, N_points_out, 0)
            
        #bin_cross_section(sigma_fine, sigma_out_log, nu_fine_1, nu_fine_2, nu_fine_3, nu_out, N_points_out, 1)
        
    t_final = time.perf_counter()
    total_final = t_final-t_start
        
    print('Total runtime: ' + str(total_final) + ' s')
        
        
def write_output_file():
    #***** Now write output files *****#
            
    if (condor_run == False): f = open(output_directory + str(species) + '_T' + str(T_arr[t]) + 'K_log_P' + str(log_P_arr[p]) + '_sigma.txt','w')
    elif (condor_run == True): f = open(output_directory + str(species) + '_T' + str(T) + 'K_log_P' + str(log_P) + '_sigma.txt','w')
                    
    for i in range(len(nu_out)):
        f.write('%.8f %.8e \n' %(nu_out[i], sigma_out[i]))
                        
    f.close()

    
def create_cross_section(input_directory, cluster_run, log_pressure, temperature, nu_min, nu_max, dnu, pressure_broadening = 'default', Voigt_cutoff, Voigt_sub_spacing, N_alpha_samples, S_cut):
    # Will need if-else for condor_run = True or False
    
    # For a single point in P-T space:
    if (cluster_run == False):
        log_P_arr = np.array([log_pressure])        # log_10 (Pressure/bar)
        T_arr = np.array([temperature])         # Temperature (K)
        N_P = len(log_P_arr)
        N_T = len(T_arr)
    
    T_pf_raw, Q_raw = load_pf(input_dir)
    
    is_molecule = check_molecule(molecule)
    
    if is_molecule and pressure_broadening == 'default':
        broadening = det_broad(input_dir)
        if broadening == 'H2-He':
            J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He = read_H2_He(input_dir)
        if broadening == 'air':
            J_max, gamma_0_air, n_L_air = read_air(input_dir)
        if broadening == 'Burrows':
            J_max, gamma_0_Burrows = read_Burrows(input_dir)
        
    if database == 'exomol':
        E, g, J = load_ExoMol(input_dir)
    
    if database == 'hitran':
        return
    
    if database == 'hitemp':
        return
    
    if database == 'vald':
        return
    
    
    # Start clock for timing program
    t_start = time.perf_counter()

    #***** Load pressure and temperature for this calculation *****#
    P_arr = np.power(10.0, log_P_arr)   

    # If running on Condor
    if (condor_run == True):
        
        idx_PT = int(sys.argv[1])
        log_P = log_P_arr[idx_PT//len(T_arr)]   # Log10 atmospheric pressure (bar)
        P = P_arr[idx_PT//len(T_arr)]   # Atmospheric pressure (bar)
        T = T_arr[idx_PT%len(T_arr)]   # Atmospheric temperature (K)
    
        N_P = 1
        N_T = 1
    
    else:
    
        N_P = len(log_P_arr)
        N_T = len(T_arr)

    for p in range(N_P):

        for t in range(N_T):
    
            if (condor_run == False):
            
                P = P_arr[p]   # Atmospheric pressure (bar)
                T = T_arr[t]   # Atmospheric temperature (K)
    
    return
