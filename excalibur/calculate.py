import numpy as np
import numba
import re

# Determine and import the correct version of numba based on its local version
#*****
version = numba.__version__
dots = re.finditer('[.]', version)
positions = [match.start() for match in dots]
version = version[:positions[1]]
version = float(version)
if version >= 0.49:
    from numba.core.decorators import jit
else:
    from numba.decorators import jit
    
from numba import prange
#*****    
    
import time
import h5py

from excalibur.constants import kb, c, c2, m_e, pi, T_ref
import excalibur.Voigt as Voigt
   

@jit(nopython=True)
def find_index(val, grid_start, grid_end, N_grid):
    
    if (val < grid_start): 
        return 0
    
    elif (val > grid_end):
        return N_grid-1
    
    else:
        i = (N_grid-1) * ((val - grid_start) / (grid_end - grid_start))
        if ((i%1)<=0.5):
            return int(i)
        else:
            return int(i)+1
        
@jit(nopython=True)
def prior_index(val, grid_start, grid_end, N_grid):
    
    if (val < grid_start): 
        return 0
    
    elif (val > grid_end):
        return N_grid-1
    
    else:
        i = (N_grid-1) * ((val - grid_start) / (grid_end - grid_start))
        return int(i)

        
@jit(nopython=True)
def compute_transition_frequencies(E, upper_state, lower_state):
        
    nu_trans = np.zeros(len(upper_state))
        
    for i in range(len(upper_state)):
            
        E_upper = E[upper_state[i]-1]   # Note: state 1 is index 0 in state file, hence the -1
        E_lower = E[lower_state[i]-1]   # Note: state 1 is index 0 in state file, hence the -1
            
        nu_trans[i] = (E_upper-E_lower)   # Transition frequency
        
    return nu_trans

@jit(nopython=True)
def compute_line_intensity_EXOMOL(A_trans, g_state, E_state, nu_0_trans, T, Q_T, upper_state, lower_state):
        
    S = np.zeros(len(upper_state))   # Line strength for transition from initial to final state
        
    for i in range(len(upper_state)):
            
        g_upper = float(g_state[upper_state[i]-1])   # Note: state 1 is index 0 in state file, hence the -1
        E_lower = E_state[lower_state[i]-1]          # Note: state 1 is index 0 in state file, hence the -1
        A = A_trans[i]
        nu_0 = nu_0_trans[i]
    
        # Note, we need to multiply c by 100 to get S in cm / molecule
        S[i] = ((g_upper*A)/(8.0*pi*(c*100.0)*nu_0*nu_0)) * np.exp(-1.0*c2*E_lower/T) * ((1.0 - np.exp(-1.0*c2*nu_0/T))/Q_T)
        
    return S

@jit(nopython=True)
def compute_line_intensity_HITRAN(S_ref, Q_T, Q_ref, T_ref, T, E_low, nu_0):
    
    S = S_ref * (Q_ref/Q_T) * np.exp(-1.0*c2*E_low*((1.0/T) - (1.0/T_ref))) * ((1.0 - np.exp(-1.0*c2*nu_0/T))/(1.0 - np.exp(-1.0*c2*nu_0/T_ref)))
        
    return S

@jit(nopython=True)
def compute_line_intensity_VALD(gf, E_low, nu_0, T, Q_T):
        
    S = np.zeros(len(nu_0))   # Line strength for transition from initial to final state
    
    e_cgs = 4.80320427e-10  # Electron charge in cgs units (statC)
    c_cgs = 100.0*c         # Speed of light in cgs units (cm)
    m_e_cgs = 1000.0*m_e    # Electron mass in cgs units (g)
        
    for i in range(len(nu_0)):
    
        # Note, we need to multiply c by 100 to get S in cm / molecule
        S[i] = ((gf[i]*pi*e_cgs**2)/(m_e_cgs*c_cgs**2)) * (1.0/Q_T) * np.exp(-1.0*c2*E_low[i]/T) * (1.0 - np.exp(-1.0*c2*nu_0[i]/T))
        
    return S
    
@jit(nopython=True)
def compute_line_intensity(A, g_upper, E_lower, nu_0, T, Q):

    # Note, we need to multiply c by 100 to get S in cm / molecule
    return ((g_upper*A)/(8.0*pi*(c*100.0)*nu_0*nu_0)) * np.exp(-1.0*c2*E_lower/T) * ((1.0 - np.exp(-1.0*c2*nu_0/T))/Q)



    
@jit(nopython=True, parallel=True)
def compute_cross_section(sigma, nu_grid, nu_0, cutoffs, S, J_lower, alpha, 
                          log_alpha, alpha_sampled, log_alpha_sampled,
                          N_Voigt, Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt, dnu_out):
    
    # Store variables that are constant acros all lines to save lookup time
    N_grid = len(nu_grid)
    nu_grid_min = nu_grid[0]
    nu_grid_max = nu_grid[-1]
    log_alpha_sampled_min = log_alpha_sampled[0]
    log_alpha_sampled_max = log_alpha_sampled[-1]
    N_log_alpha_sampled = len(log_alpha_sampled)
    
    # For each transition
    for i in prange(len(nu_0)):
        
        # Store commonly used quantities as variabls to save lookup time
        J_i = J_lower[i]
        S_i = S[i]
        nu_0_i = nu_0[i]
        
        # Find index in sampled alpha array closest to actual alpha (approximate thermal broadening)        
        idx_alpha = find_index(log_alpha[i], log_alpha_sampled_min, 
                               log_alpha_sampled_max, N_log_alpha_sampled)
        
        # Store wing cutoff for this transition
        cutoff = cutoffs[J_i,idx_alpha]
    
        # Load template Voigt function and derivatives for this gamma (J_i) and closest vaue of alpha
        Voigt_0 = Voigt_arr[J_i,idx_alpha,:]
        dV_da_0 = dV_da_arr[J_i,idx_alpha,:]
        dV_dnu_0 = dV_dnu_arr[J_i,idx_alpha,:]
        
        # Load number of template Voigt function wavenumber points and grid spacing
        dnu_Voigt_line = dnu_Voigt[J_i,idx_alpha]
            
        # Find difference between true alpha and closest pre-computed value
        d_alpha = (alpha[i] - alpha_sampled[idx_alpha])
        
        # Store grid spacing ratio between the output and pre-computed line grid
        R_nu = dnu_out / dnu_Voigt_line
            
        # Find index range on computational grid within line wing cutoff          
        idx_left = prior_index((nu_0_i - cutoff), nu_grid_min, nu_grid_max, N_grid) + 1
        idx_right = prior_index((nu_0_i + cutoff), nu_grid_min, nu_grid_max, N_grid)
        
        # Compute exact location of line core (float in output grid units) 
        core_loc = (nu_0_i - nu_grid_min)/dnu_out

        # The first index on the right wing is the core, rounded down, + 1 index
        idx_right_start = int(core_loc) + 1
        
        #***** Initiate cross section calculation at left wing cutoff *****#
        
        # Note: the cross section calculation is unpacked from a single loop
        #       into multiple regions to increase computation efficiency 
        
        # Compute exact location of left wing cutoff (float in template grid units) 
        k_ref_exact = (core_loc - idx_left)*R_nu   # R_nu maps from output grid to template grid spacing
        
        # Round to find nearest point on template grid
        k_ref = int(k_ref_exact + 0.5)
        
        # Compute wavenumber difference between true wavenumber and closest template point
        d_Delta_nu = (k_ref_exact - k_ref)*dnu_Voigt_line
       
        # 1st order Taylor expansion in alpha and Delta_nu
        sigma[idx_left] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                                   (dV_dnu_0[k_ref] * d_Delta_nu))
        
        #***** Proceed along the left wing towards core *****#
       
        # Add cross section contribution from the left wing
        for k in range(idx_left+1, idx_right_start):   
               
            # Increment k_ref_exact by the relative spacing between the k and k_ref grids
            k_ref_exact -= R_nu        # Stepping closer to the line core 

            # Round to find nearest point on template grid
            k_ref = int(k_ref_exact + 0.5)
            
            # Compute wavenumber difference between true wavenumber and closest template point
            d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line
           
            # 1st order Taylor expansion in alpha and Delta_nu
            sigma[k] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                                (dV_dnu_0[k_ref] * d_Delta_nu))

        #***** Initiate cross section calculation on the right wing *****#
        
        # Reflect once crossed into the right wing
        k_ref_exact = abs(k_ref_exact - R_nu)
            
        # Round to find nearest point on template grid
        k_ref = int(k_ref_exact + 0.5)
        
        # Compute wavenumber difference between true wavenumber and closest template point
        d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line
       
        # 1st order Taylor expansion in alpha and Delta_nu
        sigma[idx_right_start] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                                          (dV_dnu_0[k_ref] * d_Delta_nu))    
            
        #***** Proceed along the right wing towards the cutoff *****#
        
        # Add cross section contribution from the right wing
        for k in range(idx_right_start+1, idx_right+1):        # +1 to include end index
               
            # Increment k_ref_exact by the relative spacing between the k and k_ref grids
            k_ref_exact += R_nu        # Stepping away from the line core

            # Round to find nearest point on template grid
            k_ref = int(k_ref_exact + 0.5)
            
            # Compute wavenumber difference between true wavenumber and closest template point
            d_Delta_nu = (k_ref_exact - k_ref) * dnu_Voigt_line
           
            # 1st order Taylor expansion in alpha and Delta_nu
            sigma[k] += S_i * (Voigt_0[k_ref] + (dV_da_0[k_ref] * d_alpha) +
                                                (dV_dnu_0[k_ref] * d_Delta_nu))
            

@jit(nopython=True, parallel=True)
def compute_cross_section_atom(sigma, nu_grid, nu_0, S, cutoffs, N_Voigt, Voigt_arr):
    
    # Store variables that are constant acros all lines to save lookup time
    N_grid = len(nu_grid)
    nu_grid_min = nu_grid[0]
    nu_grid_max = nu_grid[-1]    
    
    for i in prange(len(nu_0)):
    
        # Store commonly used quantities as variabls to save lookup time
        N_V = N_Voigt[i]
        nu_0_i = nu_0[i]
        S_i = S[i]
        
        # Store wing cutoff for this transition
        cutoff = cutoffs[i]
    
        # Load pre-computed Voigt array for this line
        profile = Voigt_arr[i,:]
            
        # Find index range on computational grid within line wing cutoff          
        idx_left = prior_index((nu_0_i - cutoff), nu_grid_min, nu_grid_max, N_grid) + 1
        idx_right = prior_index((nu_0_i + cutoff), nu_grid_min, nu_grid_max, N_grid)
            
        # Add contribution of this line to the cross section        
        sigma[idx_left:idx_right+1] += S_i * profile[0:N_V]
    

# TBD: optimise atomic calculation time
def compute_cross_section_atom_OLD(sigma, N_grid, nu_0, nu_detune, nu_fine_start,
                                   nu_fine_end, S, T, alpha, gamma, cutoffs, 
                                   N_Voigt_points, species_ID, nu_min, nu_max):
    
    for i in range(len(nu_0)):
    
        N_V = N_Voigt_points[i]
        nu_0_i = nu_0[i]
        S_i = S[i]
            
        # Find index on fine grid of transition centre         
        idx = find_index(nu_0_i, nu_fine_start, nu_fine_end, N_grid)
            
        # Generate Voigt function array for this transition (also accounts for sub-Lorentzian resonance lines)
        profile = Voigt.Generate_Voigt_atoms(nu_0_i, nu_detune, gamma[i], alpha[i], T, cutoffs[i], N_V, species_ID)
        
        # If transition goes off left grid edge
        if ((nu_0_i-cutoffs[i]) < nu_min):
 
            # Central value
            sigma[idx] += (S_i * profile[0])
                
            for k in range(1, N_V):
                    
                # Voigt profile wings
                opac_val = (S_i * profile[k])
                    
                sigma[idx+k] += opac_val       # Forward direction
                
                if ((idx-k)>=0):
                    sigma[idx-k] += opac_val   # Backward direction 
                    
        # If transition goes off left grid edge
        elif ((nu_0_i+cutoffs[i]) > nu_max):
 
            # Central value
            sigma[idx] += (S_i * profile[0])
                
            for k in range(1, N_V):
                    
                # Voigt profile wings
                opac_val = (S_i * profile[k])
                    
                sigma[idx-k] += opac_val       # Backward direction
                
                if ((idx+k)<N_grid):
                    sigma[idx+k] += opac_val   # Forward direction 
                    
        else:
                
            # Central value
            sigma[idx] += (S_i * profile[0])
                
            for k in range(1, N_V):
                    
                # Voigt profile wings
                opac_val = (S_i * profile[k])
                    
                sigma[idx+k] += opac_val    # Forward direction
                sigma[idx-k] += opac_val    # Backward direction 
            
            

    
def cross_section_EXOMOL(linelist_files, input_directory, nu_grid, sigma, 
                         alpha_sampled, m, T, Q_T, g_arr, E_arr, J_arr, J_max, 
                         N_Voigt, cutoffs, Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt, S_cut):
    
    
    # Initialise counters for number of completed transitions
    nu_0_tot = 0
    nu_0_completed = 0
    
    # Store variables to be used in future loops
    nu_min = nu_grid[0]   # Minimum wavenumner on computational grid
    nu_max = nu_grid[-1]  # Maximum wavenumber on computational grid
    dnu_out = nu_grid[1] - nu_grid[0]
    log_alpha_sampled = np.log10(alpha_sampled)   # Array of template Doppler HWHM
    
    # Start timer for cross section computation
    t_begin_calc = time.perf_counter()
    
    # Begin loop over line list files
    for n in range(len(linelist_files)):     
      
        # Load HDF5 file containing this portion of the line list
        trans_file = h5py.File(input_directory + linelist_files[n], 'r')
    
        print('Computing transitions from ' + linelist_files[n] + ' | ' + 
              str((100.0*n/len(linelist_files))) + '% complete')
        
        # Start running timer of computation time for terminal output
        t_running = time.perf_counter()
            
        # Load upper and lower state indices and Einstein A coefficients
        upper_state = np.array(trans_file['Upper State'])
        lower_state = np.array(trans_file['Lower State'])
        A = np.power(10.0, np.array(trans_file['Log Einstein A']))
        
        # Compute transition frequencies (line core wavenumbers)
        nu_0_in = compute_transition_frequencies(E_arr, upper_state, lower_state)
    
        # Remove transitions outside computational grid
        nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]   
        upper_state = upper_state[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        lower_state = lower_state[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        A_arr = A[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        
        # Increment running counter for number of transitions
        nu_0_tot += len(nu_0)
            
        # If transitions are not in increasing wavenumber order, reorder them
        order = np.argsort(nu_0)  # Indices of nu_0 in increasing order
        nu_0 = nu_0[order]        # Apply reordering to transition wavenumbers
        upper_state = upper_state[order]  # Apply reordering to upper states
        lower_state = lower_state[order]  # Apply reordering to lower states
        A_arr = A_arr[order]              # Apply reordering to Einstein A coefficients
        
        # Store lower state J values for identifying appropriate broadening parameters
        J_low = J_arr[lower_state-1]
            
        # For J'' above the tabulated maximum, treat broadening same as the maximum J''
        J_low[np.where(J_low > J_max)] = J_max
        
        # Compute exact Doppler broadening parameters for each line  
        alpha = np.sqrt(2.0 * kb * T * np.log(2) / m) * (nu_0 / c)     
        log_alpha = np.log10(alpha)
        
        # Compute line intensities        
        S = compute_line_intensity_EXOMOL(A_arr, g_arr, E_arr, nu_0, T, Q_T, 
                                          upper_state, lower_state)
            
        # Apply intensity cutoff
        nu_0 = nu_0[np.where(S>S_cut)]
        S = S[np.where(S>S_cut)]
        
        # Delete tempory variables 
        del nu_0_in, A, A_arr, upper_state, lower_state  
        
        # Proceed if any transitions in this file satisfy the grid boundaries and intensity cutoff
        if (len(nu_0)>0): 
                   
            # Add the contributions of these lines to the cross section array (sigma)             
            compute_cross_section(sigma, nu_grid, nu_0, cutoffs, S, J_low, alpha, 
                                  log_alpha, alpha_sampled, log_alpha_sampled,
                                  N_Voigt, Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt, dnu_out)
        
        # Print time to compute transitions from this file
        t_end_running = time.perf_counter()
        total_running = t_end_running - t_running
        
        print('Completed ' + str(nu_0_tot - nu_0_completed) + 
              ' transitions in ' + str(total_running) + ' s')
        
        # Update running tally of completed transitions
        nu_0_completed = nu_0_tot   
        
        # Close HDF5 transition file
        trans_file.close()   
    
    # Print total computation time for entire line list
    t_end_calc = time.perf_counter()
    total_calc = t_end_calc - t_begin_calc

    print('Calculation complete!')
    print('Completed ' + str(nu_0_tot) + ' transitions in ' + str(total_calc) + ' s')
    

def cross_section_HITRAN(linelist_files, input_directory, nu_grid, sigma, 
                         alpha_sampled, m, T, Q_T, Q_ref, J_max, N_Voigt, 
                         cutoffs, Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt, S_cut):
    
    # Initialise counters for number of completed transitions
    nu_0_tot = 0
    nu_0_completed = 0
    
    # Store variables to be used in future loops
    nu_min = nu_grid[0]   # Minimum wavenumner on computational grid
    nu_max = nu_grid[-1]  # Maximum wavenumber on computational grid
    dnu_out = nu_grid[1] - nu_grid[0]
    log_alpha_sampled = np.log10(alpha_sampled)   # Array of template Doppler HWHM
    
    # Start timer for cross section computation
    t_begin_calc = time.perf_counter()
    
    # Begin loop over line list files
    for n in range(len(linelist_files)):      
        
        # Load HDF5 file containing this portion of the line list
        trans_file = h5py.File(input_directory + linelist_files[n], 'r')
        
        print('Computing transitions from ' + linelist_files[n] + ' | ' + 
              str((100.0*n/len(linelist_files))) + '% complete')
        
        # Start running timer for transition computation time terminal output
        t_running = time.perf_counter()
        
        # Load variables from line list
        nu_0_in = np.array(trans_file['Transition Wavenumber'])
        S_ref_in = np.power(10.0, np.array(trans_file['Log Line Intensity']))
        E_low_in = np.array(trans_file['Lower State E'])
        J_low_in = np.array(trans_file['Lower State J']).astype(np.int64)

        # Remove transitions outside computational grid
        nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]   
        S_ref = S_ref_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        E_low = E_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        J_low = J_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        
        # Increment running counter for number of transitions
        nu_0_tot += len(nu_0)
    
        # If transitions are not in increasing wavenumber order, rearrange
        order = np.argsort(nu_0)  # Indices of nu_0 in increasing order
        nu_0 = nu_0[order]        # Apply reordering to transition wavenumbers
        S_ref = S_ref[order]      # Apply reordering to reference line intensities
        E_low = E_low[order]      # Apply reordering to lower state energies
        J_low = J_low[order]      # Apply reordering to lower state J
        
        # For J'' above the tabulated maximum, treat broadening same as the maximum J''
        J_low[np.where(J_low > J_max)] = J_max
        
        # Compute exact Doppler broadening parameters for each line  
        alpha = np.sqrt(2.0 * kb * T * np.log(2) / m) * (nu_0 / c)     
        log_alpha = np.log10(alpha)
    
        # Compute line intensities        
        S = compute_line_intensity_HITRAN(S_ref, Q_T, Q_ref, T_ref, T, E_low, nu_0)
        
        # Apply intensity cutoff
        nu_0 = nu_0[np.where(S>S_cut)]
        S = S[np.where(S>S_cut)]
        
        # Delete tempory variables 
        del nu_0_in, S_ref_in, S_ref, E_low_in, E_low, J_low_in
        
        # Proceed if any transitions in this file satisfy the grid boundaries and intensity cutoff
        if (len(nu_0)>0): 
                   
            # Add the contributions of these lines to the cross section array (sigma)             
            compute_cross_section(sigma, nu_grid, nu_0, cutoffs, S, J_low, alpha, 
                                  log_alpha, alpha_sampled, log_alpha_sampled,
                                  N_Voigt, Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt, dnu_out)
        
        # Print time to compute transitions from this file
        t_end_running = time.perf_counter()
        total_running = t_end_running - t_running
        
        print('Completed ' + str(nu_0_tot - nu_0_completed) + 
              ' transitions in ' + str(total_running) + ' s')
        
        # Update running tally of completed transitions
        nu_0_completed = nu_0_tot   
        
        # Close HDF5 transition file
        trans_file.close()   
    
    # Print total computation time for entire line list
    t_end_calc = time.perf_counter()
    total_calc = t_end_calc - t_begin_calc

    print('Calculation complete!')
    print('Completed ' + str(nu_0_tot) + ' transitions in ' + str(total_calc) + ' s')

def produce_total_cross_section_VALD_atom(nu_grid, sigma, nu_0_in, 
                                          E_low, gf, m, T, Q_T, N_Voigt, cutoffs,
                                          Voigt_arr, S_cut):
    
    # Initialise counters for number of completed transitions
    nu_0_tot = 0
    
    # Store variables to be used in future loops
    nu_min = nu_grid[0]   # Minimum wavenumner on computational grid
    nu_max = nu_grid[-1]  # Maximum wavenumber on computational grid
    
    # Start timer for cross section computation
    t_begin_calc = time.perf_counter()
        
    # Remove transitions outside computational grid
    nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]  
    gf = gf[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
    E_low = E_low[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
    
    # Increment running counter for number of transitions
    nu_0_tot += len(nu_0)
        
    # Compute line intensities     
    S = compute_line_intensity_VALD(gf, E_low, nu_0, T, Q_T)
            
    # Apply intensity cutoff (disabled for atoms)
 #   nu_0 = nu_0[np.where(S>S_cut)]
 #   S = S[np.where(S>S_cut)]
        
    if (len(nu_0)>0): # If any transitions in chunk exceed intensity cutoff
        
        compute_cross_section_atom(sigma, nu_grid, nu_0, S, 
                                   cutoffs, N_Voigt, Voigt_arr)        

    # Print total computation time for entire line list
    t_end_calc = time.perf_counter()
    total_calc = t_end_calc - t_begin_calc

    print('Calculation complete!')
    print('Completed ' + str(nu_0_tot) + ' transitions in ' + str(total_calc) + ' s')
    

@jit(nopython=True)
def bin_cross_section_atom(sigma_fine, sigma_out, nu_fine_start, nu_fine_end, nu_out, N_fine, N_out, option):
    
    # To avoid logging null values
    #sigma_fine[np.where(sigma_fine <= 0.0)] = 1.0e-250 
    #log_sigma_fine = np.log10(sigma_fine)
    
    for k in range(N_out):
        
        # Locate wavenumber values corresponding to left and right bin edges
        if (k!=0) and (k!=(N_out-1)):    
            nu_l = 0.5*(nu_out[k-1] + nu_out[k])
            nu_r = 0.5*(nu_out[k] + nu_out[k+1])
        
        # Treat cases of 1st and last bin seperately (extrapolation off coarse grid)
        elif (k==0): 
            nu_l = nu_out[0] - 0.5*(nu_out[1] - nu_out[0])
            nu_r = 0.5*(nu_out[0] + nu_out[1])
        
        # Treat cases of 1st and last bin seperately (extrapolation off coarse grid)
        elif (k==(N_out-1)):
            nu_l = 0.5*(nu_out[N_out-2] + nu_out[N_out-1])
            nu_r = nu_out[k] + 0.5*(nu_out[N_out-1] - nu_out[N_out-2])
            
        # Find nearest indicies on fine grid corresponding to bin edges
        idx_l = find_index(nu_l, nu_fine_start, nu_fine_end, N_fine)   
        idx_r = find_index(nu_r, nu_fine_start, nu_fine_end, N_fine)
        
        # If averaging cross section within given bin
        if (option==0): sigma_out[k] = np.mean(sigma_fine[idx_l:(idx_r+1)])
        
        # If log-averaging cross section within given bin
        elif (option==1): sigma_out[k] = np.power(10.0, np.mean(np.log10(sigma_fine[idx_l:(idx_r+1)])))


