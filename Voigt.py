import numpy as np
import scipy.constants as sc
from scipy.special import wofz as Faddeeva
#import scipy.special as special
#import numba_special  # The import generates Numba overloads for special
from scipy.integrate import trapz, simps, quad
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

from constants import c2

#@jit(nopython=True)
def Voigt_width(gamma, alpha):
    
    return 0.5346*gamma + np.sqrt(0.2166*gamma*gamma + alpha*alpha)
    
#@jit
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
    
    for k in range(N):
        
        z = complex(x[k], y)
        W = Faddeeva(z)
        K = W.real
        L = W.imag
        
        Voigt[k] = (dim_V * K)/norm
        dV_da[k] = const1 * (b1 + (b2[k]*K) + (b3[k]*L))/norm
        dV_dv[k] = const2 * (y*L - x[k]*K)/norm                  # First derivative wrt nu is simpler
        #d2V_dv2[k] = const3 * (c1 + (c2[k]*K) + (c3[k]*L))/norm                  
    
    return Voigt, dV_da, dV_dv

#@jit
def Voigt_function(nu, gamma, alpha):
    
    x = np.sqrt(np.log(2.0)) * (nu/alpha)
    y = np.sqrt(np.log(2.0)) * (gamma/alpha)
    
    z = complex(x, y)
    
    coeff = np.sqrt(np.log(2.0)/np.pi) * (1.0/alpha)
    
    return (coeff * Faddeeva(z).real)

#@jit
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

#@jit              
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
     
#@jit
def Generate_Voigt_grid_molecules(Voigt_arr, dV_da_arr, dV_dnu_arr, gamma_arr, alpha_arr, alpha_ref, cutoffs, N):
    
    # Initialise wavenumber grids from line centre up to cutoff in each spectral region (dividers at nu_ref in config.py)
    nu1 = np.linspace(0, cutoffs[0], N[0])
    nu2 = np.linspace(0, cutoffs[1], N[1])
    nu3 = np.linspace(0, cutoffs[2], N[2])
    
    for i in range(len(gamma_arr)):    # For each gamma
        
        for j in range(len(alpha_arr)):   # For each alpha
            
            if (alpha_arr[j] <= alpha_ref[1]):
            
                # First calculate the integral of the Voigt profile out to the cutoff for normalisation purposes
                norm = 2.0*quad(Voigt_function, 0, cutoffs[0], args=(gamma_arr[i],alpha_arr[j]))[0]
            
                # Now calculate the Voigt profile and 1st derivative wrt alpha for this gamma and alpha
                Voigt_arr[i,j,0:N[0]], dV_da_arr[i,j,0:N[0]], dV_dnu_arr[i,j,0:N[0]] = Voigt_and_derivatives(nu1, gamma_arr[i], alpha_arr[j], norm)
                
            elif ((alpha_arr[j] > alpha_ref[1]) and (alpha_arr[j] <= alpha_ref[2])):
            
                # First calculate the integral of the Voigt profile out to the cutoff for normalisation purposes
                norm = 2.0*quad(Voigt_function, 0, cutoffs[1], args=(gamma_arr[i],alpha_arr[j]))[0]
            
                # Now calculate the Voigt profile and 1st derivative wrt alpha for this gamma and alpha
                Voigt_arr[i,j,0:N[1]], dV_da_arr[i,j,0:N[1]], dV_dnu_arr[i,j,0:N[1]] = Voigt_and_derivatives(nu2, gamma_arr[i], alpha_arr[j], norm)
                
            elif (alpha_arr[j] > alpha_ref[2]):
            
                # First calculate the integral of the Voigt profile out to the cutoff for normalisation purposes
                norm = 2.0*quad(Voigt_function, 0, cutoffs[2], args=(gamma_arr[i],alpha_arr[j]))[0]
            
                # Now calculate the Voigt profile and 1st derivative wrt alpha for this gamma and alpha
                Voigt_arr[i,j,0:N[2]], dV_da_arr[i,j,0:N[2]], dV_dnu_arr[i,j,0:N[2]] = Voigt_and_derivatives(nu3, gamma_arr[i], alpha_arr[j], norm)

                       
def gamma_L_VALD(gamma_vdw, m_s, broadener):
    
    ''' Computes Lorentzian HWHM at 296K and 1 bar for a given broadener
        from a tabulated VALD van der Waals broadening constant.
        
        Inputs:
            
        gamma_vdw => van der Waals parameter from VALD
        m_s => mass of species whose spectral line is being broadened (u)
        broadener => identity of broadening species (H2 or He)
        
        Outputs:
            
        gamma_L_0 => Lorentzian HWHM at reference T (296K) and P (1 bar)
        n_L => Temperature exponent (fixed to -0.7 for van der Waals theory)
        
        '''
    
    alpha_H = 0.666793  # Polarisability of atomic hydrogen (A^-3)
    m_H = 1.007825      # Mass of atomic hydrogen (u)
    
    if (broadener == 'H2'): 
        alpha_p = 0.805000  # Polarisability of molecular hydrogen (A^-3)
        m_p = 2.01565       # Mass of molecular hydrogen (u)
        
    elif (broadener == 'He'): 
        alpha_p = 0.205052  # Polarisability of helium (A^-3)
        m_p = 4.002603      # Mass of helium (u)
        
    else: print ("Invalid broadener for VALD!")
    
    # Compute Lorentzian HWHM
    gamma_L_0 = (2.2593427e7 * gamma_vdw * np.power(((m_H*(m_s+m_p))/(m_p*(m_s+m_H))), (3.0/10.0)) * 
                                           np.power((alpha_p/alpha_H), (2.0/5.0)))

    # Temperature exponent
    n_L = 0.7
    
    return gamma_L_0, n_L

def gamma_L_impact(E_low, E_up, l_low, l_up, species, m_s, broadener):
    
    ''' Computes Lorentzian HWHM at 296K and 1 bar for a given broadener
        using van der Waals impact theory.
        
        Inputs:
            
        E_low => lower level energy (cm^-1)
        E_up => upper level energy (cm^-1)
        l_low => lower level orbital angular momentum
        l_up => upper level orbital angular momentum
        species => identity of species whose spectral line is being broadened
        m_s => mass of species whose spectral line is being broadened (u)
        broadener => identity of broadening species (H2 or He)
        
        Outputs:
            
        gamma_L_0 => Lorentzian HWHM at reference T (296K) and P (1 bar)
        n_L => Temperature exponent (fixed to -0.7 for van der Waals theory)
        
        '''
    
    alpha_H = 0.666793  # Polarisability of atomic hydrogen (A^-3)
    
    E_inf_eV = {'Li': 5.3917, 'Na': 5.1391, 'K': 4.3407, 'Rb': 4.1771, 'Cs': 3.8939}    # Ionisation energy (eV)             
    E_inf = E_inf_eV[species] * 8065.547574991239  # convert from eV to cm^-1
        
    if (species in ['Li', 'Na','K', 'Cs', 'Rb']):
        Z = 0   # Ion charge
    
    if (broadener == 'H2'): 
        alpha_p = 0.805000  # Polarisability of molecular hydrogen (A^-3)
        m_p = 2.01565       # Mass of molecular hydrogen (u)
        
    elif (broadener == 'He'): 
        alpha_p = 0.205052  # Polarisability of helium (A^-3)
        m_p = 4.002603      # Mass of helium (u)
        
    else: print ("Invalid broadener for VALD!")
    
    # Evaluate effective principal quantum number for lower and upper levels
    n_low_sq = ((sc.Rydberg/100.0) * (Z + 1.0)**2)/(E_inf - E_low)
    n_up_sq =  ((sc.Rydberg/100.0) * (Z + 1.0)**2)/(E_inf - E_up)
    
    #if ((np.sqrt(n_low_sq) < (l_low+1.0)) or (np.sqrt(n_up_sq) < (l_up+1.0))):
        #print ("Quantum numbers out of range of Hydrogenic approximation!")
        #print np.sqrt(n_low_sq), (l_low+1)
        #return -1.0, -1.0
    
    # Evaluate mean square orbital radius for lower and upper levels (in Bohr radii)
    r_low_sq = (n_low_sq/(2.0*(Z+1)**2)) * (5.0*n_low_sq + 1.0 - (3.0*l_low*(l_low + 1.0)))
    r_up_sq = (n_up_sq/(2.0*(Z+1)**2)) * (5.0*n_up_sq + 1.0 - (3.0*l_up*(l_up + 1.0)))
    
    # For lines where the Hydrogenic approximation breaks down, return zero vdw line width
    if (r_up_sq < r_low_sq):
        return 0.0, 0.7   # Reference HWHM | temperature exponent (dummy in this case)
    
    # Compute Lorentzian HWHM
    gamma_L_0 = (0.1972 * np.power(((m_s+m_p)/(m_s*m_p)), (3.0/10.0)) * 
                          np.power((alpha_p/alpha_H), (2.0/5.0)) *
                          np.power((r_up_sq - r_low_sq), (2.0/5.0)))
    
    # Temperature exponent
    n_L = 0.7
    
    return gamma_L_0, n_L

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
    
#***** For old analytic Lorentzian implimentation *****#
    
# Constants for analytic Na/K Lorentzian
a_n_Na1 = np.array([1.46, 1.86, 2.16, 2.42, 2.63, 2.80])  # Na D1
a_n_Na2 = np.array([1.96, 2.49, 2.94, 3.24, 3.59, 3.87])  # Na D2
a_n_K1 = np.array([1.55, 1.96, 2.48, 2.53, 2.76, 2.97])   # K D1
a_n_K2 = np.array([2.33, 2.94, 3.49, 3.90, 4.27, 4.51])   # K D2

T_grid_NaK = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0])
NT_NaK = len(T_grid_NaK)     

@jit(nopython = True)
def prior_index(vec, value, start):
    '''Finds the index of a vector closest to a specified value'''
    
    value_tmp = value
    
    if (value_tmp > vec[-1]):
        return (len(vec) - 1)
    
    # Check if value out of bounds, if so set to edge value
    if (value_tmp < vec[0]): value_tmp = vec[0]
    if (value_tmp > vec[-2]): value_tmp = vec[-2]
    
    index = start
    
    for i in range(len(vec)-start):
        if (vec[i+start] > value_tmp): 
            index = (i+start) - 1
            break
            
    return index

#***** Analytic alkali metal prescription *****#
@jit(nopython = True)
def interpolate_NaK(T_index, T, a_n):
    
    T_1 = T_grid_NaK[T_index]
    T_2 = T_grid_NaK[T_index+1]
    
    if (T_index == (NT_NaK - 1)):
        a_1 = a_n[T_index]
        a_x = a_1
    
    else:
        a_1 = a_n[T_index]
        a_2 = a_n[T_index+1]
        a_x = a_1 + (T - T_1) * ((a_2 - a_1) / (T_2 - T_1))

    return a_x * 1.0e-26  # units: cm^-1 / m^-3      

@jit(nopython = True)
def Boltz_factor(T, nu_0):

    # Physical constants in filthy units
    #c_cgs = 2.997925e10       # Speed of light (cm s^-1)
    h_cgs = 6.626e-27         # Planck constant (cgs)
    kb_cgs = 1.381e-16        # Boltzmann constant (cgs)
    
    #wl_cgs = wl * 1.0e2   # Wavelength array in cm
    #nu = c_cgs / wl_cgs
    
    #Boltz = np.zeros(shape=(len(nu)))
    
    #for k in range(len(nu)):
            
    return np.exp((-1.0 * h_cgs * nu_0)/(kb_cgs * T))

    
#@jit(nopython = True)
def analytic_alkali(P, T, nu_bar, species):
    
    # Physical constants in filthy units
    #AMU_cgs = 1.6605655e-24  # Atomic mass unit (g])    
    e_cgs = 4.803204e-10      # Charge of electron in esu 
    me_cgs = 9.109e-28        # Electron mass (g)
    c_cgs = 2.997925e10       # Speed of light (cm s^-1)
    h_cgs = 6.626e-27         # Planck constant (cgs)
    
    nu = nu_bar*c_cgs
    
    sigma = np.zeros(shape=(len(nu)))
    B_12 = np.zeros(shape=(len(nu)))
    
    for line in range(2):
        
        if ((species == 0) and (line == 0)):  #Na D1
            
            # Line centre and oscillator strength
            wl_0, f_12 = 5896.0e-8, 0.3179
            a_n_x = a_n_Na1
            
        if ((species == 0) and (line == 1)):  #Na D2
            
            # Line centre and oscillator strength
            wl_0, f_12 = 5890.0e-8, 0.6310
            a_n_x = a_n_Na2
            
        if ((species == 1) and (line == 0)):  #K D1
            
            # Line centre and oscillator strength
            wl_0, f_12 = 7700.0e-8, 0.3390
            a_n_x = a_n_K1
            
        if ((species == 1) and (line == 1)):  #K D2
            
            # Line centre and oscillator strength
            wl_0, f_12 = 7665.0e-8, 0.6820
            a_n_x = a_n_K2
    
        nu_0 = c_cgs / wl_0
        
        Boltz = Boltz_factor(T, nu_0)

        for k in range(len(nu)):
            
            B_12[k] = ((4.0 * np.pi * np.pi * e_cgs * e_cgs)/(h_cgs * nu[k] * me_cgs * c_cgs)) * f_12    # Einstein B coefficient    
        
        # Evaluate width of Loretzian
        y = prior_index(T_grid_NaK, T, 0)     # Find previous index in T-array for each grid value
        a_n = interpolate_NaK(y, T, a_n_x)    # Evalaute interpolated a_n at each layer temperature
        a_L = (a_n * (P*1.0e5)/(sc.k * T)) * c_cgs   # Lorentz half-width (Hz) 
        #Gamma = 2.0 * a_L           # Lorentzian width (Hz)
        
        for k in range(len(nu)):
            
            # Evalaute Lorentzian line profile
            profile = (a_L / np.pi) / ((nu[k] - nu_0)*(nu[k] - nu_0) + a_L*a_L)
    
            # Evaluate cross section (Note: the += combines the two lines)  
            sigma[k] += (((h_cgs * nu[k])/(4.0 * np.pi)) * B_12[k] * 
                          (1.0 - Boltz) *
                           profile)   # Output in cm^2
            
    return sigma
    
    
    
    
    
    
    
    
    
    
    
    
    
    