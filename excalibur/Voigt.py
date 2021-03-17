import numpy as np
from scipy.special import wofz as Faddeeva
from scipy.integrate import trapz, simps, quad
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
#*****

from excalibur.constants import kb, c, c2

@jit(nopython=True)
def prior_index(val, grid_start, grid_end, N_grid):
    
    if (val < grid_start): 
        return 0
    
    elif (val > grid_end):
        return N_grid-1
    
    else:
        i = (N_grid-1) * ((val - grid_start) / (grid_end - grid_start))
        return int(i)

@jit(nopython = True)
def Voigt_HWHM(gamma_L, alpha_D):
            
    gamma_V = (0.5346 * gamma_L + np.sqrt(0.2166 * gamma_L**2 + alpha_D**2))
    
    return gamma_V

@jit(nopython = True)
def Voigt_HWHM_arr(gamma_L, alpha_D):
    
    gamma_V = np.zeros(shape=(len(gamma_L), len(alpha_D)))
    
    for i in range(len(gamma_L)):
        for j in range(len(alpha_D)):
            
            gamma_V[i,j] = (0.5346 * gamma_L[i] + np.sqrt(0.2166 * gamma_L[i]**2 + alpha_D[j]**2))
    
    return gamma_V
    

def Voigt_and_derivatives(nu, gamma, alpha, norm):
        
    N = len(nu)
    
    Voigt = np.zeros(N)    # Voigt profile
    dV_da = np.zeros(N)    # 1st derivative wrt alpha
    dV_dv = np.zeros(N)    # 1st derivative wrt wavenumber displacement from line centre (nu-nu_0)
    #d2V_dv2 = np.zeros(N)  # 2nd derivative wrt wavenumber displacement from line centre (nu-nu_0)
    
    x = np.sqrt(np.log(2.0)) * (nu/alpha)
    y = np.sqrt(np.log(2.0)) * (gamma/alpha)
    
    dim_V = np.sqrt(np.log(2.0)/np.pi) * (1.0/alpha)  # Coefficient containing dimensions of Voigt profile
    
    # Calculate coefficients for 1st derivative wrt alpha: (b1 + b2*K + b3*L)
    const1 = (2.0/(alpha**2)) * np.sqrt(np.log(2.0)/np.pi)
    const2 = (2.0/(alpha**2)) * np.sqrt((np.log(2.0))**2/np.pi)
    #const3 = (4.0/(alpha**3)) * np.sqrt((np.log(2.0))**3/np.pi)
    
    b1 = (y/np.sqrt(np.pi))
    b2 = ((x*x - y*y) - 0.5)
    b3 = (-2.0*x*y)
    
    #c1 = (y/np.sqrt(np.pi))
    #c2 = ((x*x - y*y) - 0.5)
    #c3 = (-2.0*x*y)
        
    z = np.empty(len(x), dtype=np.complex128)
    z.real = x
    z.imag = y
        
    W = Faddeeva(z)
    K = W.real
    L = W.imag
        
    Voigt = (dim_V * K)/norm
    dV_da = const1 * (b1 + (b2*K) + (b3*L))/norm
    dV_dv = const2 * (y*L - x*K)/norm             # First derivative wrt nu is simpler
    #d2V_dv2[k] = const3 * (c1 + (c2[k]*K) + (c3[k]*L))/norm                  
    
    return Voigt, dV_da, dV_dv


def Voigt_function(nu, gamma, alpha):
    
    x = np.sqrt(np.log(2.0)) * (nu/alpha)
    y = np.sqrt(np.log(2.0)) * (gamma/alpha)
    
    z = np.empty(1, dtype=np.complex128)
    z.real = x
    z.imag = y
    
    coeff = np.sqrt(np.log(2.0)/np.pi) * (1.0/alpha)
    
    return (coeff * Faddeeva(z).real)

def Voigt_function_vector(nu, gamma, alpha):
    
    x = np.sqrt(np.log(2.0)) * (nu/alpha)
    y = np.sqrt(np.log(2.0)) * (gamma/alpha)
    
    z = np.empty(len(x), dtype=np.complex128)
    z.real = x
    z.imag = y
    
    coeff = np.sqrt(np.log(2.0)/np.pi) * (1.0/alpha)
    
    return (coeff * Faddeeva(z).real)

def Voigt_function_sub_Lorentzian(nu, gamma, alpha, nu_detune, nu_F, T):
    
    coeff = np.sqrt(np.log(2.0)/np.pi) * (1.0/alpha)
    
    if (nu <= nu_detune):
        
        x = np.sqrt(np.log(2.0)) * (nu/alpha)
        y = np.sqrt(np.log(2.0)) * (gamma/alpha)
        
        z = np.empty(1, dtype=np.complex128)
        z.real = x
        z.imag = y
        
        V = (coeff * Faddeeva(z).real)
    
        return V
    
    else:
        
        x_detune = np.sqrt(np.log(2.0)) * (nu_detune/alpha)
        y = np.sqrt(np.log(2.0)) * (gamma/alpha)
        
        z_detune = np.empty(1, dtype=np.complex128)
        z_detune.real = x_detune
        z_detune.imag = y
    
        V_detune = (coeff * Faddeeva(z_detune).real)
        
        return V_detune * (nu_detune/nu)**(3.0/2.0) * np.exp((-1.0*c2*nu/T) * (nu/nu_F))

def Voigt_function_sub_Lorentzian_vector(nu, gamma, alpha, nu_detune, nu_F, T):
    
    coeff = np.sqrt(np.log(2.0)/np.pi) * (1.0/alpha)

    profile = np.zeros_like(nu)
    
    # Deal with wavenumbers less than the detuning frequency
    nu_1 = nu[abs(nu) < nu_detune]

    x_1 = np.sqrt(np.log(2.0)) * (nu_1/alpha)
    y_1 = np.sqrt(np.log(2.0)) * (gamma/alpha)
    
    z_1 = np.empty(len(x_1), dtype=np.complex128)
    z_1.real = x_1
    z_1.imag = y_1
    
    V = (coeff * Faddeeva(z_1).real)
    
    profile[abs(nu) < nu_detune] = V
        
    # Deal with wavenumbers exceeding the detuning frequency
    nu_2 = nu[abs(nu) >= nu_detune]

    x_2 = np.sqrt(np.log(2.0)) * (nu_detune/alpha)
    y_2 = np.sqrt(np.log(2.0)) * (gamma/alpha)
        
    z_2 = np.empty(1, dtype=np.complex128)
    z_2.real = x_2
    z_2.imag = y_2
    
    V_detune = (coeff * Faddeeva(z_2).real) * (nu_detune/abs(nu_2))**(3.0/2.0) * np.exp((-1.0*c2*abs(nu_2)/T) * (abs(nu_2)/nu_F))
        
    profile[abs(nu) >= nu_detune] = V_detune

    return profile 
    
          
def Generate_Voigt_atoms(nu_0, nu_detune, gamma, alpha, T, cutoff, N, species_ID):
    
    # Initialise wavenumber grid up to cutoff
    nu = np.linspace(0, cutoff, N)
    
    # Initialise output array
    Voigt_arr = np.zeros(N)
    
    nu_F = 0.0
        
    # Special treatment of line wings for alkali resonant lines (see Baudino + 2015)
    if (((species_ID == 0) and (int(nu_0) in [16978, 16960])) or       # Na D lines  (int to avoid small numerical errors)
        ((species_ID == 1) and (int(nu_0) in [13046, 12988]))):        # K D lines   (int to avoid small numerical errors)
            
        if (species_ID == 0):
            nu_F = 5000.0
        elif (species_ID == 1):
            nu_F = 1600.0
            
        # Compute renormalising factor for truncated function
        norm = 2.0*quad(Voigt_function_sub_Lorentzian, 0, cutoff, args=(gamma, alpha, nu_detune, nu_F, T))[0]  # For renormalising truncated function
        
        if (norm < 0.90): norm = 1.0  # Integration failures tend to result in values close to zero
            
        for k in range(N):   # For each wavenumber
                
            Voigt_arr[k] = Voigt_function_sub_Lorentzian(nu[k], gamma, alpha, nu_detune, nu_F, T)/norm
            
        return Voigt_arr
                
    # For non-resonant lines, simply use a Voigt function up to the cutoff
    else:
        
        # Approximate area outside cutoff for renormalisation
        norm = 0.998  #  ~ 0.998 out to +/- 500 Voigt HWHM
            
        # Compute renormalising factor for truncated function
  #      norm = 2.0*quad(Voigt_function, 0, cutoff, args=(gamma, alpha))[0]  # For renormalising truncated function
        
   #     if (norm < 0.90): norm = 1.0  # Integration failures tend to result in values close to zero
        
   #     for k in range(N):    # For each wavenumber
                
        Voigt_arr = Voigt_function_vector(nu, gamma, alpha)/norm
            
        return Voigt_arr
    
    
def Generate_Voigt_grid_atoms(Voigt_arr, atom, nu_compute, nu_0, nu_detune, nu_F, 
                              T, gamma, alpha, idx_left, idx_right, N_Voigt):
    
    
    for i in range(len(nu_0)):       # For each transition
        
        # Create wavenumber array for this line's profile (distance from line core)
        nu = nu_compute[idx_left[i]:idx_right[i]+1] - nu_0[i]
 
        # Use sub-Lorentzian wings for Na and K
        if (((atom == 'Na') and (int(nu_0[i]) in [16978, 16960])) or
            ((atom == 'K') and (int(nu_0[i]) in [13046, 12988]))): 

            # Approximate area outside cutoff for renormalisation
            norm = 0.998  #  ~ 0.998 out to +/- 500 Voigt HWHM
         
            # Compute renormalising factor for truncated function
        #    norm = 2.0*quad(Voigt_function_sub_Lorentzian, 0, cutoff, args=(gamma, alpha, nu_detune, nu_F, T))[0]  # For renormalising truncated function
        
        #    if (norm < 0.90): norm = 1.0  # Integration failures tend to result in values close to zero
                
            Voigt_arr[i,0:N_Voigt[i]] = Voigt_function_sub_Lorentzian_vector(nu, gamma[i], alpha[i], nu_detune, nu_F, T)/norm   

        # For non-resonant lines, simply use a Voigt function
        else:
            
            # Approximate area outside cutoff for renormalisation
            norm = 0.998  #  ~ 0.998 out to +/- 500 Voigt HWHM
             
             # Calculate the integral of the Voigt profile out to the cutoff for normalisation (SLOW)
    #         norm = 2.0*quad(Voigt_function, 0, cutoffs[i,j], args=(gamma_arr[i],alpha_arr[j]))[0]
             
    #        if (norm < 0.90): norm = 1.0  # Integration failures tend to result in values close to zero
    
            Voigt_arr[i,0:N_Voigt[i]] = Voigt_function_vector(nu, gamma[i], alpha[i])/norm
                 

def Generate_Voigt_grid_molecules(Voigt_arr, dV_da_arr, dV_dnu_arr, gamma_arr, 
                                  alpha_arr, cutoffs, N_Voigt):
    
    for i in range(len(gamma_arr)):       # For each Lorentzian width
        for j in range(len(alpha_arr)):   # For each Doppler width
        
            # Approximate area outside cutoff for renormalisation
            norm = 0.998  #  ~ 0.998 out to +/- 500 Voigt HWHM
            
            # Calculate the integral of the Voigt profile out to the cutoff for normalisation (SLOW)
   #         norm = 2.0*quad(Voigt_function, 0, cutoffs[i,j], args=(gamma_arr[i],alpha_arr[j]))[0]
            
            # Create wavenumber array for this template profile (line core to cutoff)
            nu = np.linspace(0.0, cutoffs[i,j], N_Voigt[i,j])
            
            # Now calculate the Voigt profile and 1st derivative wrt alpha for this gamma and alpha
            (Voigt_arr[i,j,0:N_Voigt[i,j]], 
            dV_da_arr[i,j,0:N_Voigt[i,j]], 
            dV_dnu_arr[i,j,0:N_Voigt[i,j]]) = Voigt_and_derivatives(nu, gamma_arr[i], 
                                                                    alpha_arr[j], norm)

                                                                    
def precompute_atoms(atom, nu_compute, m, T, gamma, nu_0, Voigt_cutoff, cut_max):
    
    # Calculate Voigt width for each transition
    alpha = np.sqrt(2.0*kb*T*np.log(2)/m) * (np.array(nu_0)/c)   # Doppler HWHM for each transition
    gamma_V = Voigt_HWHM(gamma, alpha)   # Voigt HWHM
    
    # Line cutoffs for each template profile at min(500 gamma_V, 30 cm^-1)    
    cutoffs = np.minimum((Voigt_cutoff * gamma_V), cut_max)
    
    # Special treatment of line wings for alkali resonant lines (see Baudino + 2015)
    if (atom == 'Na'):
        cutoffs[np.where(nu_0.astype(np.int64) == 16978)[0][0]] = 9000.0  # Cutoff at 9000 cm^-1
        cutoffs[np.where(nu_0.astype(np.int64) == 16960)[0][0]] = 9000.0  # Cutoff at 9000 cm^-1
        nu_detune = 30.0 * np.power((T/500.0), 0.6)   # Detuning requency
        nu_F = 5000.0                                 # Fit parameter
    elif (atom == 'K'):
        cutoffs[np.where(nu_0.astype(np.int64) == 13046)[0][0]] = 9000.0  # Cutoff at 9000 cm^-1
        cutoffs[np.where(nu_0.astype(np.int64) == 12988)[0][0]] = 9000.0  # Cutoff at 9000 cm^-1
        nu_detune = 20.0 * np.power((T/500.0), 0.6)   # Detuning frequency
        nu_F = 1600.0                                 # Fit parameter
    else:
        nu_detune = cut_max    # Not used for ordinary lines
        nu_F = 0.0             # Not used for ordinary lines
          
    # Initialise arrays containing computational grid indices within cutoffs
    idx_left = np.zeros(shape=len(nu_0), dtype=np.int64)
    idx_right = np.zeros(shape=len(nu_0), dtype=np.int64)

    for i in range(len(nu_0)):
        
        # Find index range on computational grid within line wing cutoff          
        idx_left[i] = prior_index((nu_0[i] - cutoffs[i]), nu_compute[0], nu_compute[-1], len(nu_compute)) + 1
        idx_right[i] = prior_index((nu_0[i] + cutoffs[i]), nu_compute[0], nu_compute[-1], len(nu_compute))
    
    # Find number of computational grid points lying within cutoffs for each line
    N_Voigt = (idx_right + 1) - idx_left

    # Initialise Voigt profiles array for all lines
    # Zeros are left for any points beyond the cutoff (local N_Voigt_nu) to preserve regular array shape
    Voigt_arr = np.empty(shape=(len(nu_0), np.max(N_Voigt)))  # dtype = np.float32

    # TBD: more efficient mememroy usage for alkali resonance lines    
 
    # Precompute Voigt profiles for each line on computational grid
    Generate_Voigt_grid_atoms(Voigt_arr, atom, nu_compute, nu_0, nu_detune, nu_F, 
                              T, gamma, alpha, idx_left, idx_right, N_Voigt)

    return cutoffs, N_Voigt, Voigt_arr


def precompute_molecules(nu_compute, dnu_out, m, T, Voigt_sub_spacing, Voigt_cutoff, 
                         N_alpha_samples, gamma_L, cut_max):
    '''
    Pre-compute Voigt profiles and derivatives, for use in the Perturbed 
    Template Voigt (PTB) method of molecular cross section computation.

    Parameters
    ----------
    nu_compute : TYPE
        DESCRIPTION.
    dnu_out : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    Voigt_sub_spacing : TYPE
        DESCRIPTION.
    Voigt_cutoff : TYPE
        DESCRIPTION.
    N_alpha_samples : TYPE
        DESCRIPTION.
    gamma_L : TYPE
        DESCRIPTION.
    cut_max : TYPE
        DESCRIPTION.

    Returns
    -------
    nu_sampled : TYPE
        DESCRIPTION.
    alpha_sampled : TYPE
        DESCRIPTION.
    cutoffs : TYPE
        DESCRIPTION.
    N_Voigt : TYPE
        DESCRIPTION.
    Voigt_arr : TYPE
        DESCRIPTION.
    dV_da_arr : TYPE
        DESCRIPTION.
    dV_dnu_arr : TYPE
        DESCRIPTION.
    dnu_Voigt : TYPE
        DESCRIPTION.

    '''
    
    # Create array of Doppler HWHM alpha to approximate true values of alpha (N=500 log-spaced => max error of 0.5%)
    nu_sampled = np.logspace(np.log10(nu_compute[0]), np.log10(nu_compute[-1]), N_alpha_samples)
    alpha_sampled = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_sampled/c)
    
    # Compute Voigt profile HWHM array for the template profiles
    gamma_V = Voigt_HWHM_arr(gamma_L, alpha_sampled)
    
    # Line cutoffs for each template profile at min(500 gamma_V, 30cm^-1)    
    cutoffs = np.minimum((Voigt_cutoff * gamma_V), cut_max)
    
    # Wavenumber spacing for each template Voigt profile (smallest of gamma_V/6 or 0.01cm^-1)
    dnu_Voigt = np.minimum((gamma_V * Voigt_sub_spacing), dnu_out)
    
    # Find number of grid points for each template Voigt profile from line core to cutoff
    N_Voigt = np.rint(cutoffs/dnu_Voigt).astype(np.int64) + 1  
    
    # Adjust dnu_Voigt slightly to match an exact integer number of grid spaces
    dnu_Voigt = cutoffs/(N_Voigt - 1)
                
    # Initialise template Voigt profiles and Voigt first derivative arrays 
    # Zeros are left for any points beyond the cutoff (local N_Voigt_nu) to preserve regular array shape
    Voigt_arr = np.zeros(shape=(len(gamma_L), len(alpha_sampled), np.max(N_Voigt)))    
    dV_da_arr = np.zeros(shape=(len(gamma_L), len(alpha_sampled), np.max(N_Voigt))) 
    dV_dnu_arr = np.zeros(shape=(len(gamma_L), len(alpha_sampled), np.max(N_Voigt)))  
    
    # Precompute template Voigt profiles (zeros left for any elements beyond cutoffs)
    Generate_Voigt_grid_molecules(Voigt_arr, dV_da_arr, dV_dnu_arr, gamma_L, 
                                  alpha_sampled, cutoffs, N_Voigt)

    return (nu_sampled, alpha_sampled, cutoffs, N_Voigt, 
            Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt)
      


    
    
    
    
    
    
    
    
    
    
    
