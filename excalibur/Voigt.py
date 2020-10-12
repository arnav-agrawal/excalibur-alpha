import numpy as np
import scipy.constants as sc
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

from excalibur.constants import kb, c, c2, nu_ref


def HWHM(gamma, alpha):
    
    return 0.5346*gamma + np.sqrt(0.2166*gamma*gamma + alpha*alpha)
    

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
    dV_dv = const2 * (y*L - x*K)/norm                  # First derivative wrt nu is simpler
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


def Voigt_function_sub_Lorentzian(nu, gamma, alpha, nu_detune, nu_F, T):
    
    coeff = np.sqrt(np.log(2.0)/np.pi) * (1.0/alpha)
    
    if (nu <= nu_detune):
        
        x = np.sqrt(np.log(2.0)) * (nu/alpha)
        y = np.sqrt(np.log(2.0)) * (gamma/alpha)
        
        z = complex(x, y)
        
        V = (coeff * Faddeeva(z).real)
    
        return V
    
    else:
        
        x_detune = np.sqrt(np.log(2.0)) * (nu_detune/alpha)
        y = np.sqrt(np.log(2.0)) * (gamma/alpha)
    
        z_detune = complex(x_detune, y)
    
        V_detune = (coeff * Faddeeva(z_detune).real)
        
        return V_detune * (nu_detune/nu)**(3.0/2.0) * np.exp((-1.0*c2*nu/T) * (nu/nu_F))


def Generate_Voigt_grid_line_by_line(Voigt_arr, nu_0, nu_detune, gamma_arr, alpha_arr, T, cutoffs, N_Voigt_points, species):
    
    for i in range(len(nu_0)):   # For each transition
    
        # Initialise wavenumber grids up to cutoff
        N = N_Voigt_points[i]
        nu = np.linspace(0, cutoffs[i], N)
        
        # Special treatment of line wings for alkali resonant lines (see Baudino + 2015)
        if (((species == 'Na') and (int(nu_0[i]) in [16978, 16960])) or
            ((species == 'K') and (int(nu_0[i]) in [13046, 12988]))):
            
            if   (species == 'Na'): nu_F = 5000.0
            elif (species == 'K'):  nu_F = 1600.0
            
            # Compute renormalising factor for truncated function
            norm = 2.0*quad(Voigt_function_sub_Lorentzian, 0, cutoffs[i], args=(gamma_arr[i], alpha_arr[i], nu_detune, nu_F, T))[0]  # For renormalising truncated function
                
            for k in range(N):   # For each wavenumber
                
                Voigt_arr[i,k] = Voigt_function_sub_Lorentzian(nu[k], gamma_arr[i], alpha_arr[i], nu_detune, nu_F, T)/norm
                
        # For non-resonant lines, simply use a Voigt function up to the cutoff
        else:
            
            # Compute renormalising factor for truncated function
            norm = 2.0*quad(Voigt_function, 0, cutoffs[i], args=(gamma_arr[i], alpha_arr[i]))[0]  #
            
            for k in range(N):    # For each wavenumber
                
                Voigt_arr[i,k] = Voigt_function(nu[k], gamma_arr[i], alpha_arr[i])/norm

             
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
            
        # Compute renormalising factor for truncated function
        norm = 2.0*quad(Voigt_function, 0, cutoff, args=(gamma, alpha))[0]  # For renormalising truncated function
        
        if (norm < 0.90): norm = 1.0  # Integration failures tend to result in values close to zero
        
        for k in range(N):    # For each wavenumber
                
            Voigt_arr[k] = Voigt_function(nu[k], gamma, alpha)/norm
            
        return Voigt_arr
     

def Generate_Voigt_grid_molecules(Voigt_arr, dV_da_arr, dV_dnu_arr, gamma_arr, alpha_arr, alpha_ref, cutoffs, N):
    
    # Initialise wavenumber grids from line centre up to cutoff in each spectral region (dividers at nu_ref in config.py)
    nu1 = np.linspace(0, cutoffs[0], N[0])
    nu2 = np.linspace(0, cutoffs[1], N[1])
    nu3 = np.linspace(0, cutoffs[2], N[2])
    
    for i in range(len(gamma_arr)):    # For each gamma
        
        for j in range(len(alpha_arr)):   # For each alpha
        
            norm = 0.998   # Approximation of area under Voigt profile (out to +/- 500 HWHM) for renormalisation
            
            if (alpha_arr[j] <= alpha_ref[1]):
            
                # First calculate the integral of the Voigt profile out to the cutoff for normalisation purposes
      #          norm = 2.0*quad(Voigt_function, 0, cutoffs[0], args=(gamma_arr[i],alpha_arr[j]))[0]
            
                # Now calculate the Voigt profile and 1st derivative wrt alpha for this gamma and alpha
                Voigt_arr[i,j,0:N[0]], dV_da_arr[i,j,0:N[0]], dV_dnu_arr[i,j,0:N[0]] = Voigt_and_derivatives(nu1, gamma_arr[i], alpha_arr[j], norm)
                
            elif ((alpha_arr[j] > alpha_ref[1]) and (alpha_arr[j] <= alpha_ref[2])):
            
                # First calculate the integral of the Voigt profile out to the cutoff for normalisation purposes
      #          norm = 2.0*quad(Voigt_function, 0, cutoffs[1], args=(gamma_arr[i],alpha_arr[j]))[0]
            
                # Now calculate the Voigt profile and 1st derivative wrt alpha for this gamma and alpha
                Voigt_arr[i,j,0:N[1]], dV_da_arr[i,j,0:N[1]], dV_dnu_arr[i,j,0:N[1]] = Voigt_and_derivatives(nu2, gamma_arr[i], alpha_arr[j], norm)
                
            elif (alpha_arr[j] > alpha_ref[2]):
            
                # First calculate the integral of the Voigt profile out to the cutoff for normalisation purposes
      #          norm = 2.0*quad(Voigt_function, 0, cutoffs[2], args=(gamma_arr[i],alpha_arr[j]))[0]
            
                # Now calculate the Voigt profile and 1st derivative wrt alpha for this gamma and alpha
                Voigt_arr[i,j,0:N[2]], dV_da_arr[i,j,0:N[2]], dV_dnu_arr[i,j,0:N[2]] = Voigt_and_derivatives(nu3, gamma_arr[i], alpha_arr[j], norm)
                

       
def precompute(nu_max, N_alpha_samples, T, m, cutoffs, dnu_fine, gamma, alpha_ref):
    '''
    Pre-compute Voigt profiles and derivatives, for use in the Generalised 
    Vectorised Voigt (GVV) method of molecular cross section computation.

    Parameters
    ----------
    nu_max : TYPE
        DESCRIPTION.
    N_alpha_samples : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    cutoffs : TYPE
        DESCRIPTION.
    dnu_fine : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    alpha_ref : TYPE
        DESCRIPTION.

    Returns
    -------
    nu_sampled : TYPE
        DESCRIPTION.
    alpha_sampled : TYPE
        DESCRIPTION.
    Voigt_arr : TYPE
        DESCRIPTION.
    dV_da_arr : TYPE
        DESCRIPTION.
    dV_dnu_arr : TYPE
        DESCRIPTION.
    N_Voigt_points : TYPE
        DESCRIPTION.

    '''
        
    # First, create an array of values of alpha to approximate true values of alpha (N=500 log-spaced => max error of 0.5%)
    nu_sampled = np.logspace(np.log10(nu_ref[0]), np.log10(nu_max), N_alpha_samples)
    alpha_sampled = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_sampled/c)
    
    # Evaluate number of frequency points for each Voigt function in each spectral region - up to cutoff @ min(500 gamma_V, 30cm^-1)
    N_Voigt_points = ((cutoffs/dnu_fine).astype(np.int64)) + 1  
            
    # Pre-compute and store Voigt functions and first derivatives wrt alpha 
    Voigt_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))    # For H2O: V(51,500,3001)
    dV_da_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))    # For H2O: V(51,500,3001)
    dV_dnu_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))   # For H2O: V(51,500,3001)
    
    Generate_Voigt_grid_molecules(Voigt_arr, dV_da_arr, dV_dnu_arr, gamma, 
                                  alpha_sampled, alpha_ref, cutoffs, N_Voigt_points)

    return nu_sampled, alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr, N_Voigt_points
        
                     


#Voigt_approx = Voigt_arr[0,50,:] + (dV_da_arr[0,50,:])*(alpha_sampled[51]-alpha_sampled[50])
    
#plt.semilogy(nu, Voigt_arr[0,50,:], lw=0.1, label='50')
#plt.semilogy(nu, Voigt_arr[0,51,:], lw=0.1, label='51 true')
#plt.semilogy(nu, Voigt_approx[:], lw=0.1, label='51 approx')
#plt.xlim([0.0, 0.01])
#plt.ylim([1.0e0, 1.0e3])

#legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':7})
            
#plt.savefig('Voigt_test.pdf')

#plt.semilogy(nu, err)
#plt.savefig('test_err.pdf')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
