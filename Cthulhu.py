# CTHULU MAIN ROUTINE
# V0.98 (natural broadening added for atoms)

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline as Interp
import matplotlib.pyplot as plt
from numba.decorators import jit
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d as gauss_conv
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter
import time
import h5py
import sys
import os

from Voigt import Voigt_width, Generate_Voigt_grid_molecules, gamma_L_VALD, gamma_L_impact, analytic_alkali
from calculations import find_index, prior_index, bin_cross_section_atom, bin_cross_section_molecule
from calculations import produce_total_cross_section_EXOMOL, produce_total_cross_section_HITRAN
from calculations import produce_total_cross_section_VALD_atom, produce_total_cross_section_VALD_molecule

from config import input_directory, output_directory, file_format, linelist, linelist_files, species, broadening
from config import calculation_type, compute_Voigt, make_cross_section, plot_results, write_output, condor_run
#***** First load in partition function, states, and broadening files *****#

print('Processing ' + linelist + ' line list for ' + species)

if (file_format == 'EXOMOL'): 
    
    print("File format is EXOMOL")
    
    # Read in states file (EXOMOL only)
    states_file_name = [filename for filename in os.listdir(input_directory) if filename.endswith('.states')]
    states_file = pd.read_csv(input_directory + states_file_name[0], sep = '\s+', header=None)
    E = np.array(states_file[1])
    g = np.array(states_file[2])
    J = np.array(states_file[3]).astype(np.int64)
    
    del states_file  # Delete file to free up memory
    
elif (file_format == 'HITRAN'): 
    
    print("File format is HITRAN")
    
    # No files beyond standard .par file and .pf file required for HITRAN
    
elif (file_format == 'VALD'):
    
    print("File format is VALD")
    
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
    
else:
    print("Invalid linelist format!")
    sys.exit()
    
# Read in partition function file (same format for EXOMOL, HITRAN, and VALD)
pf_file_name = [filename for filename in os.listdir(input_directory) if filename.endswith('.pf')]
pf_file = pd.read_csv(input_directory + pf_file_name[0], sep= ' ', header=None, skiprows=1)
T_pf_raw = np.array(pf_file[0]).astype(np.float64)
Q_raw = np.array(pf_file[1])

del pf_file   # Delete file to free up memory

# For molecules, read in EXOMOL style broadening files (handled seperately for each line for atoms)
if (calculation_type == 'molecule'):
        
    if (broadening == 'H2-He'):

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
            
    elif (broadening == 'air'):
        
        # Read in air broadening file
        broad_file_air = pd.read_csv(input_directory + 'air.broad', sep = ' ', header=None, skiprows=1)
        J_max = int(np.max(np.array(broad_file_air[0])))
        gamma_0_air = np.array(broad_file_air[1])
        n_L_air = np.array(broad_file_air[2])
        
        del broad_file_air   # Delete file to free up memory
        
    elif (broadening == 'Burrows'):
        
        # Read in air broadening file
        broad_file_Burrows = pd.read_csv(input_directory + 'Burrows.broad', sep = ' ', header=None, skiprows=1)
        J_max = int(np.max(np.array(broad_file_Burrows[0])))
        gamma_0_Burrows = np.array(broad_file_Burrows[1])
        n_L_Burrows = np.array(broad_file_Burrows[2])       # Not really needed, as temperature exponent = 0 for all J''
        
        del broad_file_Burrows   # Delete file to free up memory
        
    elif (broadening == 'fixed'):
        from config import gamma_0, n_L    # Load default values
        J_max = 0
    
    else:
        print("Invalid broadening choice!")
        sys.exit()

# Load in various necessary variables defined in config.py
from config import nu_min, nu_max, nu_out_min, nu_out_max, nu_ref, dnu_out
from config import u, c, kb, log_P_arr, T_arr, cut_max
from config import species_id, masses, T_ref, P_ref, X_H2, X_He
from config import Voigt_sub_spacing, N_alpha_samples, Voigt_cutoff
    
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
    
        #***** Interpolate (and extrapolate) partition function to a fine grid *****#
        pf_spline = Interp(T_pf_raw, Q_raw, k=5)
        T_pf_fine = np.linspace(1.0, 10000.0, 9999)       # New temperature grid (extrapolated to 4000K)
        Q_fine = pf_spline(T_pf_fine)                    # Extrapolated partition function
        i_T = np.argmin(np.abs(T_pf_fine - T))           # Index of partition function temperature array closest to desired temperature
        i_T_ref = np.argmin(np.abs(T_pf_fine - T_ref))   # Index of partition function temperature array closest to reference temperature
        Q_T = Q_fine[i_T]                                # Partition function at given temperature, Q(T)
        Q_T_ref = Q_fine[i_T_ref]                        # Partition function at reference temperature, Q(T_ref)
            
        #***** Compute pressure broadening parameters *****#
        m = masses[species_id]*u    # Species mass (kg)
        
        if (calculation_type == 'atom'):
            
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
        
        # Using either pre-loaded broadening parameters (molecules) or calculated (atoms), compute Lorentzian HWHM
        if (broadening == 'H2-He'):
            gamma = (gamma_0_H2 * np.power((T_ref/T), n_L_H2) * (P/P_ref) * X_H2 +   # H2+He Lorentzian HWHM for given T, P, and J (ang. mom.)
                     gamma_0_He * np.power((T_ref/T), n_L_He) * (P/P_ref) * X_He)    # Note that these are only a function of J''
        elif (broadening == 'air'):
            gamma = (gamma_0_air * np.power((T_ref/T), n_L_air) * (P/P_ref))      # Air-broadened Lorentzian HWHM for given T, P, and J (ang. mom.)
        elif (broadening == 'Burrows'):
            gamma = (gamma_0_Burrows * (P/P_ref))      # Equation (15) in Sharp & Burrows (2007)     
        elif (broadening == 'fixed'):
            gamma = np.array([(gamma_0 * np.power((T_ref/T), n_L) * (P/P_ref))])  # Fixed Lorentizian HWHM (1 element array)
            
        if (calculation_type == 'atom'):
            gamma += ((1.0/(4.0*np.pi*(100.0*c))) * gamma_nat)  # Add natural line widths
            
        #***** Create wavelength grid *****#
        
        if (calculation_type == 'atom'):   # Simple case with a single, uniformly-spaced wavenumber grid
            
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
            
        elif (calculation_type == 'molecule'):  # Use three uniformly-spaced wavenumber grids for molecules
            
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

        print('Pre-computations complete.')
        
        #***** Begin main program *****#
        
        # Make cross section
        if (make_cross_section == True):
            
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
        
        #***** Now write output files *****#
        
        if (write_output == True):
            
            if (condor_run == False): f = open(output_directory + str(species) + '_T' + str(T_arr[t]) + 'K_log_P' + str(log_P_arr[p]) + '_sigma.txt','w')
            elif (condor_run == True): f = open(output_directory + str(species) + '_T' + str(T) + 'K_log_P' + str(log_P) + '_sigma.txt','w')
                    
            for i in range(len(nu_out)):
                f.write('%.8f %.8e \n' %(nu_out[i], sigma_out[i]))
                        
            f.close()
        
#***** Now plot some results *****#

if(plot_results == True):
    
    wl_out = 1.0e4/nu_out
    
    # Make wavenumber plot
    plt.figure()
    plt.clf()
    ax = plt.gca()
    
    plt.semilogy(nu_out, sigma_out, lw=0.3, alpha = 0.5, color = 'red', label = (species + r'Cross Section (out)'))   
    
    plt.xlim([200.0, 25000.0])
    plt.ylim([1.0e-30, 1.0e-12])

    plt.ylabel(r'Cross Section (cm$^2$)')
    plt.xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)')

    legend = plt.legend(loc='upper left', shadow=False, frameon=False, prop={'size':6})
    
    plt.savefig('../output/' + species + '_' + str(T) + 'K_' + str(P) + 'bar_nu.pdf')
    
    plt.close()
    
    # Make wavelength plot
    plt.clf()
    ax = plt.gca()
    
    xmajorLocator   = MultipleLocator(0.2)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.02)
    
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    
    plt.loglog(wl_out, sigma_out, lw=0.3, alpha = 0.5, color= 'red', label = (species + r'$\mathrm{\, \, Cross \, \, Section}$')) 
    
    plt.ylim([1.0e-30, 1.0e-14])
    plt.xlim([0.4, 10.0])
    
    plt.ylabel(r'Cross Section (cm$^2$)')
    plt.xlabel(r'Wavelength (μm)')
    
    ax.text(0.7, 5.0e-16, (r'T = ' + str(T) + r'K, P = ' + str(P) + r'bar'), fontsize = 10)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':7})
    
    #plt.show()

    plt.savefig('../output/' + species + '_' + str(T) + 'K_' + str(P) + 'bar.pdf')
    




