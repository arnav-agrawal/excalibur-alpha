import numpy as np
import pandas as pd
import numba
import re

# Determine and import the correct version of numba based on its local version
version = numba.__version__
dots = re.finditer('[.]', version)
positions = [match.start() for match in dots]
version = version[:positions[1]]
version = float(version)
if version >= 0.49:
    from numba.core.decorators import jit
else:
    from numba.decorators import jit
    
    
#from math import exp
import time
import h5py

from constants import kb, c, c2, m_e, pi, T_ref
from excalibur.Voigt import Generate_Voigt_atoms
   

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

@jit(nopython=True)
def increment_sigma(sigma, nu_0, nu_grid_start, nu_grid_end, S_i, Voigt_0, dV_da_0, d_alpha, N_grid, N_V):
                
    idx = find_index(nu_0, nu_grid_start, nu_grid_end, N_grid)
                
    sigma[idx] += (S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha)))
                
    for j in range(1, N_V):
                    
        # 1st order Taylor expansion in alpha
        opac_val = (S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha)))
                    
        sigma[idx+j] += opac_val    # Forward direction
        sigma[idx-j] += opac_val    # Backward direction

@jit(nopython=True)
def compute_cross_section_single_grid(sigma, nu_grid_start, nu_grid_end, 
                                      N_grid, nu_0, nu_ref, N_V, S, 
                                      J_lower, alpha, alpha_sampled, 
                                      nu_sampled, log_nu_sampled,
                                      log_alpha, log_alpha_sampled,
                                      Voigt_arr, dV_da_arr, dV_dnu_arr):
    
    N_alpha_samples = len(alpha_sampled)
        
    idx_boundary_1 = prior_index(np.log10(nu_ref[1]), log_nu_sampled[0], log_nu_sampled[-1], N_alpha_samples)
    idx_boundary_2 = prior_index(np.log10(nu_ref[2]), log_nu_sampled[0], log_nu_sampled[-1], N_alpha_samples)
        
    nu_ref_1_r = nu_sampled[idx_boundary_1+1]
    nu_ref_2_r = nu_sampled[idx_boundary_2+1]
    
    for i in range(len(nu_0)):
    
        nu_0_i = nu_0[i]
        S_i = S[i]
        J_i = J_lower[i]
        alpha_i = alpha[i]
        log_alpha_i = log_alpha[i]
            
        # Find index in sampled alpha array closest to actual alpha (approximate thermal broadening)        
        idx_alpha = prior_index(log_alpha_i, log_alpha_sampled[0], log_alpha_sampled[-1], N_alpha_samples)
        if   ((nu_0_i > nu_ref[1]) and (nu_0_i < nu_ref_1_r)): idx_alpha += 1
        elif ((nu_0_i > nu_ref[2]) and (nu_0_i < nu_ref_2_r)): idx_alpha += 1
            
        # Find index on fine grid of transition centre         
        idx = find_index(nu_0_i, nu_grid_start, nu_grid_end, N_grid)
            
        # Load pre-computed Voigt function and derivatives for this gamma (J_i) and closest vaue of alpha
        Voigt_0 = Voigt_arr[J_i,idx_alpha,:]
        dV_da_0 = dV_da_arr[J_i,idx_alpha,:]
            
        # Difference between true value of alpha and closest pre-computed value
        d_alpha = (alpha_i - alpha_sampled[idx_alpha])
 
        # 1st order Taylor expansion in alpha
        sigma[idx] += S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha))
                
        for j in range(1, N_V):
                    
            # 1st order Taylor expansion in alpha
            opac_val = S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha))
                    
            sigma[idx+j] += opac_val    # Forward direction
            sigma[idx-j] += opac_val    # Backward direction

#@jit(nopython=True)     
#@jit
def compute_cross_section_atom(sigma, N_grid, nu_0, nu_detune, nu_fine_start,
                               nu_fine_end, S, T, alpha, gamma, cutoffs, 
                               N_Voigt_points, species_ID, nu_min, nu_max):
    
    for i in range(len(nu_0)):
    
        N_V = N_Voigt_points[i]
        nu_0_i = nu_0[i]
        S_i = S[i]
            
        # Find index on fine grid of transition centre         
        idx = find_index(nu_0_i, nu_fine_start, nu_fine_end, N_grid)
            
        # Generate Voigt function array for this transition (also accounts for sub-Lorentzian resonance lines)
        Voigt = Generate_Voigt_atoms(nu_0_i, nu_detune, gamma[i], alpha[i], T, cutoffs[i], N_V, species_ID)
        
        # If transition goes off left grid edge
        if ((nu_0_i-cutoffs[i]) < nu_min):
 
            # Central value
            sigma[idx] += (S_i * Voigt[0])
                
            for k in range(1, N_V):
                    
                # Voigt profile wings
                opac_val = (S_i * Voigt[k])
                    
                sigma[idx+k] += opac_val       # Forward direction
                
                if ((idx-k)>=0):
                    sigma[idx-k] += opac_val   # Backward direction 
                    
        # If transition goes off left grid edge
        elif ((nu_0_i+cutoffs[i]) > nu_max):
 
            # Central value
            sigma[idx] += (S_i * Voigt[0])
                
            for k in range(1, N_V):
                    
                # Voigt profile wings
                opac_val = (S_i * Voigt[k])
                    
                sigma[idx-k] += opac_val       # Backward direction
                
                if ((idx+k)<N_grid):
                    sigma[idx+k] += opac_val   # Forward direction 
                    
        else:
                
            # Central value
            sigma[idx] += (S_i * Voigt[0])
                
            for k in range(1, N_V):
                    
                # Voigt profile wings
                opac_val = (S_i * Voigt[k])
                    
                sigma[idx+k] += opac_val    # Forward direction
                sigma[idx-k] += opac_val    # Backward direction 
            
#@jit
@jit(nopython=True)
def compute_cross_section_multiple_grids(sigma_1, sigma_2, sigma_3, 
                                         nu_fine_1_start, nu_fine_1_end,
                                         nu_fine_2_start, nu_fine_2_end,
                                         nu_fine_3_start, nu_fine_3_end,
                                         N_grid_1, N_grid_2, N_grid_3,
                                         nu_0, nu_ref, N_V_1, N_V_2, N_V_3,
                                         S, J_lower, alpha, alpha_sampled,
                                         log_alpha, log_alpha_sampled,
                                         nu_sampled, log_nu_sampled, R_21, R_32,
                                         dnu_1, dnu_2, dnu_3, cutoffs,
                                         Voigt_arr, dV_da_arr, dV_dnu_arr,
                                         nu_min, nu_max):
    
    N_alpha_samples = len(alpha_sampled)
    
    idx_boundary_1 = prior_index(np.log10(nu_ref[1]), log_nu_sampled[0], log_nu_sampled[-1], N_alpha_samples)
    idx_boundary_2 = prior_index(np.log10(nu_ref[2]), log_nu_sampled[0], log_nu_sampled[-1], N_alpha_samples)
        
    nu_ref_1_r = nu_sampled[idx_boundary_1+1]
        
    nu_ref_2_r = nu_sampled[idx_boundary_2+1]
        
    for i in range(len(nu_0)):
    
        nu_0_i = nu_0[i]
        S_i = S[i]
        J_i = J_lower[i]
        alpha_i = alpha[i]
        log_alpha_i = log_alpha[i]
        
        case = 0  # By default (if none of the below conditions apply), have a simple uniform grid with no boundary changes
            
        # If transition on 1st grid
        if ((nu_0_i > nu_min) and (nu_0_i < nu_ref[1])):
            
            case = 10                                  # Default
            
            if ((nu_0_i-cutoffs[0]) < nu_min):         # If lower Voigt wing falls off grid LHS => CASE 11
            
                case = 11
            
            elif ((nu_0_i+cutoffs[0]) > nu_ref[1]):    # If upper Voigt wing crosses into 2nd grid => CASE 12
            
                case = 12
                
            idx = find_index(nu_0_i, nu_fine_1_start, nu_fine_1_end, N_grid_1)
        
        # If transition on 2nd grid
        elif ((nu_0_i > nu_ref[1]) and (nu_0_i < nu_ref[2])): 
            
            case = 20                                  # Default
            
            if ((nu_0_i-cutoffs[1]) < nu_ref[1]):      # If lower Voigt wing crosses into 1st grid => CASE 21
        
                case = 21
            
            elif ((nu_0_i+cutoffs[1]) > nu_ref[2]):    # If upper Voigt wing crosses into 3rd grid => CASE 22
            
                case = 22
                
            idx = find_index(nu_0_i, nu_fine_2_start, nu_fine_2_end, N_grid_2)
           
        # If transition on 3rd grid
        elif ((nu_0_i > nu_ref[2]) and (nu_0_i < nu_max)): 
            
            case = 30                                  # Default
                        
            if ((nu_0_i-cutoffs[2]) < nu_ref[2]):      # If lower Voigt wing crosses into 2nd grid => CASE 31
            
                case = 31
            
            elif ((nu_0_i+cutoffs[2]) > nu_max):       # If upper Voigt wing falls off grid RHS => CASE 32
            
                case = 32
                
            idx = find_index(nu_0_i, nu_fine_3_start, nu_fine_3_end, N_grid_3)
        
        # Find index in sampled alpha array closest to actual alpha (approximate thermal broadening)        
        idx_alpha = prior_index(log_alpha_i, log_alpha_sampled[0], log_alpha_sampled[-1], N_alpha_samples)
        if   ((nu_0_i > nu_ref[1]) and (nu_0_i < nu_ref_1_r)): idx_alpha += 1
        elif ((nu_0_i > nu_ref[2]) and (nu_0_i < nu_ref_2_r)): idx_alpha += 1
        
        # Load pre-computed Voigt function and derivatives for this gamma (J_i) and closest vaue of alpha
        Voigt_0 = Voigt_arr[J_i,idx_alpha,:]
        dV_da_0 = dV_da_arr[J_i,idx_alpha,:]
        dV_dnu_0 = dV_dnu_arr[J_i,idx_alpha,:]
        
        # Difference between true value of alpha and closest pre-computed value
        d_alpha = (alpha_i - alpha_sampled[idx_alpha])
            
        # If no boundaries are encountered by upper or lower wings => CASE 10/20/30
        if   (case==10): increment_sigma(sigma_1, nu_0_i, nu_fine_1_start, nu_fine_1_end, S_i, Voigt_0, dV_da_0, d_alpha, N_grid_1, N_V_1)
        elif (case==20): increment_sigma(sigma_2, nu_0_i, nu_fine_2_start, nu_fine_2_end, S_i, Voigt_0, dV_da_0, d_alpha, N_grid_2, N_V_2)
        elif (case==30): increment_sigma(sigma_3, nu_0_i, nu_fine_3_start, nu_fine_3_end, S_i, Voigt_0, dV_da_0, d_alpha, N_grid_3, N_V_3)

        # If lower Voigt wing falls off grid LHS => CASE 11
        elif (case==11):
            
            # 1st order Taylor expansion in alpha
            sigma_1[idx] += S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha))
            
            for j in range(1, N_V_1):
                
                # 1st order Taylor expansion in alpha
                opac_val_1 = S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha))
                
                sigma_1[idx+j] += opac_val_1        # Forward direction
                
                if ((idx-j)>=0): 
                    sigma_1[idx-j] += opac_val_1    # Backward direction
                    
        # If upper Voigt wing crosses into 2nd grid => CASE 12
        elif (case==12):
            
            # Find highest value of j residing on 1st grid
            j_max = (N_grid_1-1) - idx
            
            # 1st order Taylor expansion in alpha for central value
            sigma_1[idx] += (S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha)))
            
            for j in range(1, N_V_1):
                
                # 1st order Taylor expansion in alpha
                opac_val_1 = (S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha))) 
                
                # Backwards direction is simple, as lies entirely on 1st grid
                sigma_1[idx-j] += opac_val_1      # Backward direction
                
                # Whilst upper wing is on 1st grid
                if (j<=j_max):
                
                    sigma_1[idx+j] += opac_val_1  # Forward direction
                   
                # When upper wing extends onto 2nd grid
                elif (j>j_max):
                    
                    #d_delta_nu = 0.0
                    
                    # Shift index to 2nd grid
                    j_2 = j-(j_max+1)
                    
                    # Find index multiple on 1st grid cloest in value to the true index value j_2
                    j_1 = int(j_2*R_21)
                    j_extended = (j_max+1) + j_1
                    
                    # Find true wavenumber seperation from line centre
                    #delta_nu_true = (nu_ref[1] + float(j_2*dnu_2) - nu_0_i)
                    
                    # Find closest wavenumber seperation from line centre stored in pre-computed Voigt profile
                    #delta_nu_pre = (nu_ref[1] + float(j_1*dnu_1) - nu_0_i)
                    
                    # Find difference between true and closest pre-computed wavenumber seperation for Taylor expansion
                    #d_delta_nu = (delta_nu_true - delta_nu_pre)
                    d_delta_nu = (float(j_2*dnu_2) - float(j_1*dnu_1))
                    
                    # Check that the true wavenumber seperation is still less than the cutoff for the lower wing
                    #if (delta_nu_true < cutoffs[0]):
                    if (j_extended < N_V_1):
                        
                        # 1st order Taylor expansion in alpha and delta_nu
                        opac_val_2 = (S_i * (Voigt_0[j_extended] + (dV_da_0[j_extended] * d_alpha) + (dV_dnu_0[j_extended] * d_delta_nu)))

                        sigma_2[j_2] += opac_val_2

        # If lower Voigt wing crosses into 1st grid => CASE 21
        elif (case==21):
            
            # For backwards extrapolation, we need to count how many bins have already been accounted for
            k_0 = 0
            
            # Find highest value of j residing on 2nd grid
            j_max = idx
            
            # 1st order Taylor expansion in alpha for central value
            sigma_2[idx] += (S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha)))
            
            for j in range(1, N_V_2):
                
                # 1st order Taylor expansion in alpha
                opac_val_2 = (S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha))) 
                
                # Forwards direction is simple, as lies entirely on 2nd grid
                sigma_2[idx+j] += opac_val_2      # Forward direction
                
                # Whilst lower wing is on 2nd grid
                if (j<=j_max):
                
                    sigma_2[idx-j] += opac_val_2  # Backward direction
                   
                # When lower wing extends onto 1st grid
                elif (j>j_max):
                    
                    # Shift grid 2 index to ennumerate number of equivalent spaces traversed onto 1st grid
                    j_2 = j-j_max            
                    
                    # Find number of bins in 1st grid between RHS of j_2 and nu_ref boundary
                    k_max = int(j_2*R_21)
                    
                    # Find bin index that is ~ closer to j_2-1 (right) than j_2 (left)                    
                    k_right = (k_0 + k_max)/2
                    
                    # Find closest wavenumber seperation from line centre stored in pre-computed Voigt profile
                    delta_nu_pre_right = (nu_0_i - (nu_ref[1] - (float(j_2-1)*dnu_2)))
                    delta_nu_pre_left  = (nu_0_i - (nu_ref[1] - (float(j_2)*dnu_2))) 
                     
                    for k in range(k_0, k_right):
                    
                        # Find true wavenumber seperation from line centre
                        delta_nu_true = (nu_0_i - (nu_ref[1] - float((k+1)*dnu_1)))
                    
                        # Find difference between true and closest pre-computed wavenumber seperation for Taylor expansion
                        d_delta_nu_r = (delta_nu_true - delta_nu_pre_right)    # (>0)
                        
                        # 1st order Taylor expansion in alpha and delta_nu
                        opac_val_1 = (S_i * (Voigt_0[j-1] + (dV_da_0[j-1] * d_alpha) + (dV_dnu_0[j-1] * d_delta_nu_r)))

                        sigma_1[-1-k] += opac_val_1
                        
                    for k in range(k_right, k_max):
                    
                        # Find true wavenumber seperation from line centre
                        delta_nu_true = (nu_0_i - (nu_ref[1] - float((k+1)*dnu_1)))
                    
                        # Find difference between true and closest pre-computed wavenumber seperation for Taylor expansion
                        d_delta_nu_l = (delta_nu_true - delta_nu_pre_left)     # (<0)
                        
                        # 1st order Taylor expansion in alpha and delta_nu
                        opac_val_1 = (S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha) + (dV_dnu_0[j] * d_delta_nu_l)))

                        sigma_1[-1-k] += opac_val_1
                    
                    # Finally, update completed bins counter
                    k_0 = k_max
                    
        # If upper Voigt wing crosses into 3rd grid => CASE 22
        elif (case==22):
            
            # Find highest value of j residing on 2nd grid
            j_max = (N_grid_2-1) - idx
            
            # 1st order Taylor expansion in alpha for central value
            sigma_2[idx] += (S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha)))
            
            for j in range(1, N_V_2):
                
                # 1st order Taylor expansion in alpha
                opac_val_2 = (S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha))) 
                
                # Backwards direction is simple, as lies entirely on 2nd grid
                sigma_2[idx-j] += opac_val_2      # Backward direction
                
                # Whilst upper wing is on 2nd grid
                if (j<=j_max):
                
                    sigma_2[idx+j] += opac_val_2  # Forward direction
                   
                # When upper wing extends onto 3rd grid
                elif (j>j_max):
                    
                    # Shift index to 3rd grid
                    j_3 = j-(j_max+1)
                    
                    # Find index multiple on 2nd grid cloest in value to the true index value j_3
                    j_2 = int(j_3*R_32)     
                    j_extended = (j_max+1) + j_2  # +1 is due to j_3 = 0 being exactly one grid 2 space to right
                    
                    # Find true wavenumber seperation from line centre
                    #delta_nu_true = (nu_ref[2] + float(j_3*dnu_3) - nu_0_i)
                    
                    # Find closest wavenumber seperation from line centre stored in pre-computed Voigt profile
                    #delta_nu_pre = (nu_ref[2] + float(j_2*dnu_2) - nu_0_i)
                    
                    # Find difference between true and closest pre-computed wavenumber seperation for Taylor expansion
                    d_delta_nu = (float(j_3*dnu_3) - float(j_2*dnu_2))
                    
                    # Check that the true wavenumber seperation is still less than the cutoff for the lower wing
                    #if (delta_nu_true < cutoffs[1]):
                    if (j_extended < N_V_2):
                        
                        # 1st order Taylor expansion in alpha and delta_nu
                        opac_val_3 = (S_i * (Voigt_0[j_extended] + (dV_da_0[j_extended] * d_alpha) + (dV_dnu_0[j_extended] * d_delta_nu)))

                        sigma_3[j_3] += opac_val_3
                        
        # If lower Voigt wing crosses into 2nd grid => CASE 31
        elif (case==31):
            
            # For backwards extrapolation, we need to count how many bins have already been accounted for
            k_0 = 0
            
            # Find highest value of j residing on 3rd grid
            j_max = idx
            
            # 1st order Taylor expansion in alpha for central value
            sigma_3[idx] += (S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha)))
            
            for j in range(1, N_V_3):
                
                # 1st order Taylor expansion in alpha
                opac_val_3 = (S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha))) 
                
                # Forwards direction is simple, as lies entirely on 3rd grid
                sigma_3[idx+j] += opac_val_3      # Forward direction
                
                # Whilst lower wing is on 3rd grid
                if (j<=j_max):
                
                    sigma_3[idx-j] += opac_val_3  # Backward direction
                   
                # When lower wing extends onto 2nd grid
                elif (j>j_max):
                    
                    # Shift grid 3 index to ennumerate number of equivalent spaces traversed onto 2nd grid
                    j_3 = j-j_max            
                    
                    # Find number of bins in 2nd grid between RHS of j_3 and nu_ref boundary
                    k_max = int(j_3*R_32)
                    
                    # Find bin index that is ~ closer to j_3-1 (right) than j_3 (left)                    
                    k_right = (k_0 + k_max)/2
                    
                    # Find closest wavenumber seperation from line centre stored in pre-computed Voigt profile
                    delta_nu_pre_right = (nu_0_i - (nu_ref[2] - (float(j_3-1)*dnu_3)))
                    delta_nu_pre_left  = (nu_0_i - (nu_ref[2] - (float(j_3)*dnu_3))) 
                    
                    for k in range(k_0, k_right):
                    
                        # Find true wavenumber seperation from line centre
                        delta_nu_true = (nu_0_i - (nu_ref[2] - float((k+1)*dnu_2)))
                    
                        # Find difference between true and closest pre-computed wavenumber seperation for Taylor expansion
                        d_delta_nu_r = (delta_nu_true - delta_nu_pre_right)    # (>0)
                        
                        # 1st order Taylor expansion in alpha and delta_nu
                        opac_val_2 = (S_i * (Voigt_0[j-1] + (dV_da_0[j-1] * d_alpha) + (dV_dnu_0[j-1] * d_delta_nu_r)))

                        sigma_2[-1-k] += opac_val_2
                        
                    for k in range(k_right, k_max):
                    
                        # Find true wavenumber seperation from line centre
                        delta_nu_true = (nu_0_i - (nu_ref[2] - float((k+1)*dnu_2)))
                    
                        # Find difference between true and closest pre-computed wavenumber seperation for Taylor expansion
                        d_delta_nu_l = (delta_nu_true - delta_nu_pre_left)     # (<0)
                        
                        # 1st order Taylor expansion in alpha and delta_nu
                        opac_val_2 = (S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha) + (dV_dnu_0[j] * d_delta_nu_l)))

                        sigma_2[-1-k] += opac_val_2
                    
                    # Finally, update completed bins counter
                    k_0 = k_max
                    
        # If upper Voigt wing falls off grid RHS => CASE 32
        elif (case==32):
            
            # 1st order Taylor expansion in alpha
            sigma_3[idx] += S_i * (Voigt_0[0] + (dV_da_0[0] * d_alpha))
            
            for j in range(1, N_V_3):
                
                # 1st order Taylor expansion in alpha
                opac_val_3 = S_i * (Voigt_0[j] + (dV_da_0[j] * d_alpha))
                
                sigma_3[idx-j] += opac_val_3        # Backward direction
                
                if ((idx+j)<N_grid_3): 
                    sigma_3[idx+j] += opac_val_3    # Forward direction  
                    

#@jit
def produce_total_cross_section_EXOMOL(linelist_files, input_directory, 
                                       sigma_fine, nu_sampled, nu_ref, m, T, Q_T,
                                       N_grid_1, N_grid_2, N_grid_3, dnu_fine, 
                                       N_Voigt_points, cutoffs, g_arr, E_arr, J_arr, 
                                       J_max, alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr,
                                       nu_min, nu_max, S_cut):
    
    nu_fine_1_start = nu_min
    nu_fine_1_end = (nu_ref[1] - dnu_fine[0])
    nu_fine_2_start = nu_ref[1]
    nu_fine_2_end = (nu_ref[2] - dnu_fine[1])
    nu_fine_3_start = nu_ref[2]
    nu_fine_3_end = nu_max
    
    nu_0_tot = 0
    nu_0_completed = 0
    
    log_alpha_sampled = np.log10(alpha_sampled)
    log_nu_sampled = np.log10(nu_sampled)
    
    # Number of points on each pre-computed Voigt function
    N_V_1 = N_Voigt_points[0]
    N_V_2 = N_Voigt_points[1]
    N_V_3 = N_Voigt_points[2]
    
    # Split sigma into components on each spectral grid        
    sigma_1 = sigma_fine[0:N_grid_1]
    sigma_2 = sigma_fine[N_grid_1:(N_grid_1+N_grid_2)]
    sigma_3 = sigma_fine[(N_grid_1+N_grid_2):(N_grid_1+N_grid_2+N_grid_3)]
        
    # Wavenumber spacings on each spectral grid
    dnu_1 = dnu_fine[0]
    dnu_2 = dnu_fine[1]
    dnu_3 = dnu_fine[2]
        
    # Ratios of wavenumber spacings between each grid
    R_21 = dnu_2/dnu_1
    R_32 = dnu_3/dnu_2
    
    # Start timer for cross section computation
    t_begin_calc = time.perf_counter()
    
    for n in range(len(linelist_files)):   # For each transition file
      
        trans_file = h5py.File(input_directory + linelist_files[n], 'r')
    
        print('Computing transitions from ' + linelist_files[n] + ' | ' + str((100.0*n/len(linelist_files))) + '% complete')
        
        # Start running timer for transition computation time terminal output
        t_running = time.perf_counter()
            
        upper_state = np.array(trans_file['Upper State'])
        lower_state = np.array(trans_file['Lower State'])
        A = np.power(10.0, np.array(trans_file['Log Einstein A']))
        
        #***** Compute transition frequencies *****#
        nu_0_in = compute_transition_frequencies(E_arr, upper_state, lower_state)
    
        # Remove transitions outside computational grid
        nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]   
        upper_state = upper_state[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        lower_state = lower_state[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        A_arr = A[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
            
        del nu_0_in, A   # No longer need to store these
            
        # If transitions are not in increasing wavenumber order, rearrange
        order = np.argsort(nu_0)  # Indices of nu_0 in increasing order
        nu_0 = nu_0[order]
        upper_state = upper_state[order]
        lower_state = lower_state[order]
        A_arr = A_arr[order]
        
        # Store lower state J values for identifying appropriate broadening parameters
        J_lower = J_arr[lower_state-1]
            
        # If a given J'' does not have known broadening parameters, treat transition with parameters of maximum J''
        J_lower[np.where(J_lower > J_max)] = J_max
        
        if (len(nu_0)>0): nu_0_tot += len(nu_0)
        
        #***** Compute line broadening parameters *****#    
        alpha = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_0/c)     # Doppler HWHM for each transition
        log_alpha = np.log10(alpha)
        
        #***** Compute line intensities *****#        
        S = compute_line_intensity_EXOMOL(A_arr, g_arr, E_arr, nu_0, T, Q_T, upper_state, lower_state)
            
        #nu_0 = nu_0[np.where(S>S_cut)]
        #S = S[np.where(S>S_cut)]
        
        del A_arr, upper_state, lower_state   # No longer need to store these
        
        if (len(nu_0)>0): # If any transitions exceed intensity cutoff
                
            # Read comment on RHS for an explanation of this ominous condition-->
            if (((nu_0[0]-cutoffs[0]) > nu_min) and ((nu_0[-1]+cutoffs[0]) < nu_ref[1])): 
                
                compute_cross_section_single_grid(sigma_1, nu_fine_1_start, nu_fine_1_end,    # Check if the given range of nu0 +/- cutoffs lies entirely within one of the
                                                  N_grid_1, nu_0, nu_ref, N_V_1, S,           # three pre-defined spectral ranges. If so, then can speed up cross section
                                                  J_lower, alpha, alpha_sampled,              # computation by not checking whether line wings cross grid boundaries
                                                  nu_sampled, log_nu_sampled,
                                                  log_alpha, log_alpha_sampled,
                                                  Voigt_arr, dV_da_arr, dV_dnu_arr)
                
            elif (((nu_0[0]-cutoffs[1]) > nu_ref[1]) and ((nu_0[-1]+cutoffs[1]) < nu_ref[2])): 
                
                compute_cross_section_single_grid(sigma_2, nu_fine_2_start, nu_fine_2_end, 
                                                  N_grid_2, nu_0, nu_ref, N_V_2, S,
                                                  J_lower, alpha, alpha_sampled,
                                                  nu_sampled, log_nu_sampled,
                                                  log_alpha, log_alpha_sampled,
                                                  Voigt_arr, dV_da_arr, dV_dnu_arr)
    
            elif (((nu_0[0]-cutoffs[2]) > nu_ref[2]) and ((nu_0[-1]+cutoffs[2]) < nu_max)): 
                
                compute_cross_section_single_grid(sigma_3, nu_fine_3_start, nu_fine_3_end, 
                                                  N_grid_3, nu_0, nu_ref, N_V_3, S,
                                                  J_lower, alpha, alpha_sampled,
                                                  nu_sampled, log_nu_sampled,
                                                  log_alpha, log_alpha_sampled,
                                                  Voigt_arr, dV_da_arr, dV_dnu_arr)
    
            else: 
                
                compute_cross_section_multiple_grids(sigma_1, sigma_2, sigma_3, 
                                                     nu_fine_1_start, nu_fine_1_end,
                                                     nu_fine_2_start, nu_fine_2_end,
                                                     nu_fine_3_start, nu_fine_3_end,
                                                     N_grid_1, N_grid_2, N_grid_3,
                                                     nu_0, nu_ref, N_V_1, N_V_2, N_V_3,
                                                     S, J_lower, alpha, alpha_sampled,
                                                     log_alpha, log_alpha_sampled,
                                                     nu_sampled, log_nu_sampled, R_21, R_32,
                                                     dnu_1, dnu_2, dnu_3, cutoffs,
                                                     Voigt_arr, dV_da_arr, dV_dnu_arr,
                                                     nu_min, nu_max)
        
        # Print time to compute transitions for this .trans file
        t_end_running = time.perf_counter()
        total_running = t_end_running - t_running
        
        print('Completed ' + str(nu_0_tot - nu_0_completed) + ' transitions in ' + str(total_running) + ' s')
        
        nu_0_completed = nu_0_tot   # Update running tally of completed transitions
        
        trans_file.close()   # Close HDF5 transition file
    
    # Print total computation time for entire line list
    t_end_calc = time.perf_counter()
    total_calc = t_end_calc-t_begin_calc

    print('Calculation complete!')
    print('Completed ' + str(nu_0_tot) + ' transitions in ' + str(total_calc) + ' s')


def produce_total_cross_section_HITRAN(linelist_files, input_directory, sigma_fine, 
                                       nu_sampled, nu_ref, m, T, Q_T, Q_ref,
                                       N_grid_1, N_grid_2, N_grid_3, dnu_fine, 
                                       N_Voigt_points, cutoffs, J_max, alpha_sampled,
                                       Voigt_arr, dV_da_arr, dV_dnu_arr,
                                       nu_min, nu_max, S_cut):
    
    nu_fine_1_start = nu_min
    nu_fine_1_end = (nu_ref[1] - dnu_fine[0])
    nu_fine_2_start = nu_ref[1]
    nu_fine_2_end = (nu_ref[2] - dnu_fine[1])
    nu_fine_3_start = nu_ref[2]
    nu_fine_3_end = nu_max
    
    nu_0_tot = 0
    nu_0_completed = 0
    
    log_alpha_sampled = np.log10(alpha_sampled)
    log_nu_sampled = np.log10(nu_sampled)
    
    # Number of points on each pre-computed Voigt function
    N_V_1 = N_Voigt_points[0]
    N_V_2 = N_Voigt_points[1]
    N_V_3 = N_Voigt_points[2]
    
    # Split sigma into components on each spectral grid        
    sigma_1 = sigma_fine[0:N_grid_1]
    sigma_2 = sigma_fine[N_grid_1:(N_grid_1+N_grid_2)]
    sigma_3 = sigma_fine[(N_grid_1+N_grid_2):(N_grid_1+N_grid_2+N_grid_3)]
        
    # Wavenumber spacings on each spectral grid
    dnu_1 = dnu_fine[0]
    dnu_2 = dnu_fine[1]
    dnu_3 = dnu_fine[2]
        
    # Ratios of wavenumber spacings between each grid
    R_21 = dnu_2/dnu_1
    R_32 = dnu_3/dnu_2
    
    # Start timer for cross section computation
    t_begin_calc = time.perf_counter()
    
    for n in range(len(linelist_files)):   # For each transition file
      
        trans_file = h5py.File(input_directory + linelist_files[n], 'r')
        
        print('Computing transitions from ' + linelist_files[n] + ' | ' + str((100.0*n/len(linelist_files))) + '% complete')
        
        # Start running timer for transition computation time terminal output
        t_running = time.perf_counter()
        
        #***** Load variables from linelist *****#
        nu_0_in = np.array(trans_file['Transition Wavenumber'])
        S_ref_in = np.power(10.0, np.array(trans_file['Log Line Intensity']))
        E_low_in = np.array(trans_file['Lower State E'])
        J_low_in = np.array(trans_file['Lower State J']).astype(np.int64)

        # Remove transitions outside computational grid
        nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]   
        S_ref = S_ref_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        E_low = E_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        J_low = J_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
    
        del nu_0_in, S_ref_in, E_low_in, J_low_in  # No longer need to store these
        
        # If transitions are not in increasing wavenumber order, rearrange
        order = np.argsort(nu_0)  # Indices of nu_0 in increasing order
        nu_0 = nu_0[order]
        S_ref = S_ref[order]
        E_low = E_low[order]
        J_low = J_low[order]
        
        # If a given J'' does not have known broadening parameters, treat transition with parameters of maximum J''
        J_low[np.where(J_low > J_max)] = J_max
    
        if (len(nu_0)>0): nu_0_tot += len(nu_0)
        
        #***** Compute line broadening parameters *****#    
        alpha = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_0/c)     # Doppler HWHM for each transition
        log_alpha = np.log10(alpha)
    
        #***** Compute line intensities *****#        
        S = compute_line_intensity_HITRAN(S_ref, Q_T, Q_ref, T_ref, T, E_low, nu_0)
        
        #nu_0 = nu_0[np.where(S>S_cut)]
        #S = S[np.where(S>S_cut)]
    
        del S_ref, E_low   # No longer need to store these
    
        if (len(nu_0)>0): # If any transitions exceed intensity cutoff
            
            # Read comment on RHS for an explanation of this ominous condition-->
            if (((nu_0[0]-cutoffs[0]) > nu_min) and ((nu_0[-1]+cutoffs[0]) < nu_ref[1])): 
            
                compute_cross_section_single_grid(sigma_1, nu_fine_1_start, nu_fine_1_end,    # Check if the given range of nu0 +/- cutoffs lies entirely within one of the
                                                  N_grid_1, nu_0, nu_ref, N_V_1, S,           # three pre-defined spectral ranges. If so, then can speed up cross section
                                                  J_low, alpha, alpha_sampled,                # computation by not checking whether line wings cross grid boundaries
                                                  nu_sampled, log_nu_sampled,
                                                  log_alpha, log_alpha_sampled,
                                                  Voigt_arr, dV_da_arr, dV_dnu_arr)
            
            elif (((nu_0[0]-cutoffs[1]) > nu_ref[1]) and ((nu_0[-1]+cutoffs[1]) < nu_ref[2])): 
            
                compute_cross_section_single_grid(sigma_2, nu_fine_2_start, nu_fine_2_end, 
                                                  N_grid_2, nu_0, nu_ref, N_V_2, S,
                                                  J_low, alpha, alpha_sampled,
                                                  nu_sampled, log_nu_sampled,
                                                  log_alpha, log_alpha_sampled,
                                                  Voigt_arr, dV_da_arr, dV_dnu_arr)

            elif (((nu_0[0]-cutoffs[2]) > nu_ref[2]) and ((nu_0[-1]+cutoffs[2]) < nu_max)): 
            
                compute_cross_section_single_grid(sigma_3, nu_fine_3_start, nu_fine_3_end, 
                                                  N_grid_3, nu_0, nu_ref, N_V_3, S,
                                                  J_low, alpha, alpha_sampled,
                                                  nu_sampled, log_nu_sampled,
                                                  log_alpha, log_alpha_sampled,
                                                  Voigt_arr, dV_da_arr, dV_dnu_arr)

            else: 
            
                compute_cross_section_multiple_grids(sigma_1, sigma_2, sigma_3, 
                                                     nu_fine_1_start, nu_fine_1_end,
                                                     nu_fine_2_start, nu_fine_2_end,
                                                     nu_fine_3_start, nu_fine_3_end,
                                                     N_grid_1, N_grid_2, N_grid_3,
                                                     nu_0, nu_ref, N_V_1, N_V_2, N_V_3,
                                                     S, J_low, alpha, alpha_sampled,
                                                     log_alpha, log_alpha_sampled,
                                                     nu_sampled, log_nu_sampled, R_21, R_32,
                                                     dnu_1, dnu_2, dnu_3, cutoffs,
                                                     Voigt_arr, dV_da_arr, dV_dnu_arr,
                                                     nu_min, nu_max)
            
        # Print time to compute transitions for this .trans file
        t_end_running = time.perf_counter()
        total_running = t_end_running - t_running
        
        print('Completed ' + str(nu_0_tot - nu_0_completed) + ' transitions in ' + str(total_running) + ' s')
        
        nu_0_completed = nu_0_tot   # Update running tally of completed transitions
        
        trans_file.close()   # Close HDF5 transition file
    
    # Print total computation time for entire line list
    t_end_calc = time.perf_counter()
    total_calc = t_end_calc-t_begin_calc

    print('Calculation complete!')
    print('Completed ' + str(nu_0_tot) + ' transitions in ' + str(total_calc) + ' s')
    
def produce_total_cross_section_VALD_molecule(sigma_fine, nu_sampled, nu_ref, nu_0_in,
                                              E_low_in, J_low_in, gf_in, m, T, Q_T,
                                              N_grid_1, N_grid_2, N_grid_3, dnu_fine, 
                                              N_Voigt_points, cutoffs, J_max, alpha_sampled,
                                              Voigt_arr, dV_da_arr, dV_dnu_arr,
                                              nu_min, nu_max, S_cut, species):
    
    nu_fine_1_start = nu_min
    nu_fine_1_end = (nu_ref[1] - dnu_fine[0])
    nu_fine_2_start = nu_ref[1]
    nu_fine_2_end = (nu_ref[2] - dnu_fine[1])
    nu_fine_3_start = nu_ref[2]
    nu_fine_3_end = nu_max
    
    nu_0_tot = 0
    
    log_alpha_sampled = np.log10(alpha_sampled)
    log_nu_sampled = np.log10(nu_sampled)
    
    # Number of points on each pre-computed Voigt function
    N_V_1 = N_Voigt_points[0]
    N_V_2 = N_Voigt_points[1]
    N_V_3 = N_Voigt_points[2]
    
    # Split sigma into components on each spectral grid        
    sigma_1 = sigma_fine[0:N_grid_1]
    sigma_2 = sigma_fine[N_grid_1:(N_grid_1+N_grid_2)]
    sigma_3 = sigma_fine[(N_grid_1+N_grid_2):(N_grid_1+N_grid_2+N_grid_3)]
        
    # Wavenumber spacings on each spectral grid
    dnu_1 = dnu_fine[0]
    dnu_2 = dnu_fine[1]
    dnu_3 = dnu_fine[2]
        
    # Ratios of wavenumber spacings between each grid
    R_21 = dnu_2/dnu_1
    R_32 = dnu_3/dnu_2
    
    t_begin_calc = time.clock()
        
    print('Computing transitions for ' + species)
    
    # Remove transitions outside computational grid
    nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]  
    gf = gf_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
    E_low = E_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
    J_low = J_low_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
    
    del nu_0_in, gf_in, E_low_in, J_low_in
            
    # If a given J'' does not have known broadening parameters, treat transition with parameters of maximum J''
    J_low[np.where(J_low > J_max)] = J_max
        
    if (len(nu_0)>0): nu_0_tot += len(nu_0)
        
    #***** Compute line broadening parameters *****#    
    alpha = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_0/c)     # Doppler HWHM for each transition
    log_alpha = np.log10(alpha)
    
    #***** Compute line intensities *****#        
    S = compute_line_intensity_VALD(gf, E_low, nu_0, T, Q_T)
               
    #nu_0 = nu_0[np.where(S>S_cut)]
    #S = S[np.where(S>S_cut)]
        
    if (len(nu_0)>0): # If any transitions in chunk exceed intensity cutoff
        
        # Read comment on RHS for an explanation of this ominous condition-->
        if (((nu_0[0]-cutoffs[0]) > nu_min) and ((nu_0[-1]+cutoffs[0]) < nu_ref[1])): 
                
            compute_cross_section_single_grid(sigma_1, nu_fine_1_start, nu_fine_1_end,    # Check if the given range of nu0 +/- cutoffs lies entirely within one of the
                                              N_grid_1, nu_0, nu_ref, N_V_1, S,           # three pre-defined spectral ranges. If so, then can speed up cross section
                                              J_low, alpha, alpha_sampled,                # computation by not checking whether line wings cross grid boundaries
                                              nu_sampled, log_nu_sampled,
                                              log_alpha, log_alpha_sampled,
                                              Voigt_arr, dV_da_arr, dV_dnu_arr)
                
        elif (((nu_0[0]-cutoffs[1]) > nu_ref[1]) and ((nu_0[-1]+cutoffs[1]) < nu_ref[2])): 
                
            compute_cross_section_single_grid(sigma_2, nu_fine_2_start, nu_fine_2_end, 
                                              N_grid_2, nu_0, nu_ref, N_V_2, S,
                                              J_low, alpha, alpha_sampled,
                                              nu_sampled, log_nu_sampled,
                                              log_alpha, log_alpha_sampled,
                                              Voigt_arr, dV_da_arr, dV_dnu_arr)
    
        elif (((nu_0[0]-cutoffs[2]) > nu_ref[2]) and ((nu_0[-1]+cutoffs[2]) < nu_max)): 
                
            compute_cross_section_single_grid(sigma_3, nu_fine_3_start, nu_fine_3_end, 
                                              N_grid_3, nu_0, nu_ref, N_V_3, S,
                                              J_low, alpha, alpha_sampled,
                                              nu_sampled, log_nu_sampled,
                                              log_alpha, log_alpha_sampled,
                                              Voigt_arr, dV_da_arr, dV_dnu_arr)
    
        else: 
                
            compute_cross_section_multiple_grids(sigma_1, sigma_2, sigma_3, 
                                                 nu_fine_1_start, nu_fine_1_end,
                                                 nu_fine_2_start, nu_fine_2_end,
                                                 nu_fine_3_start, nu_fine_3_end,
                                                 N_grid_1, N_grid_2, N_grid_3,
                                                 nu_0, nu_ref, N_V_1, N_V_2, N_V_3,
                                                 S, J_low, alpha, alpha_sampled,
                                                 log_alpha, log_alpha_sampled,
                                                 nu_sampled, log_nu_sampled, R_21, R_32,
                                                 dnu_1, dnu_2, dnu_3, cutoffs,
                                                 Voigt_arr, dV_da_arr, dV_dnu_arr,
                                                 nu_min, nu_max)
    
    t_end_calc = time.clock()
    total_calc = t_end_calc-t_begin_calc

    print('Calculation complete!')
    print('Completed ' + str(nu_0_tot) + ' transitions in ' + str(total_calc) + ' s')    


def produce_total_cross_section_VALD_atom(sigma_fine, nu_0_in, nu_detune, 
                                          E_low, gf, m, T, Q_T, N_points_fine,
                                          N_Voigt_points, alpha, gamma, cutoffs,
                                          nu_min, nu_max, S_cut, species):
    
    if   (species == 'Na'): species_ID = 0  # Flag for sub-Lorentizan treatment
    elif (species == 'K'):  species_ID = 1  # Flag for sub-Lorentizan treatment
    else: species_ID = -1    # Else, just treat as Voigt profile
    
    nu_fine_start = nu_min
    nu_fine_end = nu_max
    
    nu_0_tot = 0
        
    t_begin_calc = time.clock()
        
    print('Computing transitions for ' + species)
    
    # Remove transitions outside computational grid
    nu_0 = nu_0_in[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]  
    gf = gf[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
    E_low = E_low[np.where((nu_0_in >= nu_min) & (nu_0_in <= nu_max))]
        
    if (len(nu_0)>0): nu_0_tot += len(nu_0)
        
    #***** Compute line intensities *****#        
    S = compute_line_intensity_VALD(gf, E_low, nu_0, T, Q_T)
            
    #nu_0 = nu_0[np.where(S>S_cut)]
    #S = S[np.where(S>S_cut)]
        
    if (len(nu_0)>0): # If any transitions in chunk exceed intensity cutoff
        
        compute_cross_section_atom(sigma_fine, N_points_fine, nu_0, nu_detune, 
                                   nu_fine_start, nu_fine_end, S, T, alpha, 
                                   gamma, cutoffs, N_Voigt_points, species_ID,
                                   nu_min, nu_max)
    
    t_end_calc = time.clock()
    total_calc = t_end_calc-t_begin_calc

    print('Calculation complete!')
    print('Completed ' + str(nu_0_tot) + ' transitions in ' + str(total_calc) + ' s')
    

@jit(nopython=True)
def bin_cross_section_molecule(sigma_fine, sigma_out, N_grid_1, N_grid_2, N_grid_3, nu_ref,
                               nu_fine_1_start, nu_fine_1_end, nu_fine_2_start, nu_fine_2_end, 
                               nu_fine_3_start, nu_fine_3_end, nu_out, N_out, option,
                               nu_min, nu_max):
    
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
        if   ((nu_min < nu_l)    and (nu_l < nu_ref[1])): idx_l = find_index(nu_l, nu_fine_1_start, nu_fine_1_end, N_grid_1)
        elif ((nu_ref[1] < nu_l) and (nu_l < nu_ref[2])): idx_l = N_grid_1 + find_index(nu_l, nu_fine_2_start, nu_fine_2_end, N_grid_2)
        elif ((nu_ref[2] < nu_l) and (nu_l < nu_max)):    idx_l = N_grid_1 + N_grid_2 + find_index(nu_l, nu_fine_3_start, nu_fine_3_end, N_grid_3)
             
        if   ((nu_min < nu_r)    and (nu_r < nu_ref[1])): idx_r = find_index(nu_r, nu_fine_1_start, nu_fine_1_end, N_grid_1)
        elif ((nu_ref[1] < nu_r) and (nu_r < nu_ref[2])): idx_r = N_grid_1 + find_index(nu_r, nu_fine_2_start, nu_fine_2_end, N_grid_2)
        elif ((nu_ref[2] < nu_r) and (nu_r < nu_max)):    idx_r = N_grid_1 + N_grid_2 + find_index(nu_r, nu_fine_3_start, nu_fine_3_end, N_grid_3)
        
        # If averaging cross section within given bin
        if (option==0): sigma_out[k] = np.mean(sigma_fine[idx_l:(idx_r+1)])
        
        # If log-averaging cross section within given bin
        elif (option==1): sigma_out[k] = np.power(10.0, np.mean(np.log10(sigma_fine[idx_l:(idx_r+1)])))
        
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











