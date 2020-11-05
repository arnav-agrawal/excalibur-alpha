import os
import numpy as np
import pandas as pd

from excalibur.constants import Ryd, u

def det_broad(input_directory):
    '''
    Determine the type of broadening that should be used in the case that the user specifies
    'default' broadening. Order of preference is: 1) H2-He, 2) air, 3) SB07

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    broadening : String
        The type of broadening being used.

    '''

    # Molecular hydrogen + helium broadening is first choice
    if 'H2.broad' in os.listdir(input_directory) and 'He.broad' in os.listdir(input_directory):
        broadening = 'H2-He'

    # If no H2 + He boradening files, search for an air boradening file
    elif 'air.broad' in os.listdir(input_directory):
        broadening = 'air'

    # If neither of the above are available (e.g. for most metal oxides), fall back to Sharp & Burrows (2007)
    else:
        broadening = 'SB07'
        if not 'SB07.broad' in os.listdir(input_directory):
            create_SB07(input_directory)

    return broadening


def create_SB07(input_directory):
    '''
    Create a broadening file according to Eq. 15 of Sharp & Burrows (2007),
    and add it to the input_directory.

    Note: S&B (2007) state Eq. 15 gives the FWHM. Personal communication from
    Richard Freedman indicates the equation actually gives the HWHM.

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    SB07_file = input_directory + 'SB07.broad'

    # Initialise total arrays
    J = np.arange(31.0)          # Total angular momentum
    gamma_L_0 = np.zeros(31)     # Lorentizian HWHM at P_ref and T_ref
    n_L = np.zeros(31)           # Temperature exponent

    # Implement Eq. 15 from S&B07 (wihout division by 2, as already HWHM)
    for i in range(len(J)):
        gamma_L_0[i] = (0.1 - min(J[i], 30) * 0.002) / 1.01325

    # Write broadening output file
    f_out = open(SB07_file, 'w')

    f_out.write('J | gamma_L_0 | n_L \n')

    for i in range(len(J)):
        f_out.write('%.1f %.4f %.3f \n' %(J[i], gamma_L_0[i], n_L[i]))

    f_out.close()


def read_H2_He(input_directory):
    '''
    Read the H2 and He broadening files from the input directory

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    J_max : TYPE
        DESCRIPTION.
    gamma_0_H2 : TYPE
        DESCRIPTION.
    n_L_H2 : TYPE
        DESCRIPTION.
    gamma_0_He : TYPE
        DESCRIPTION.
    n_L_He : TYPE
        DESCRIPTION.

    '''

    # Read in H2 broadening file
    broad_file_H2 = pd.read_csv(input_directory + 'H2.broad',
                                sep = ' ', header=None, skiprows=1)
    J_max_H2 = int(np.max(np.array(broad_file_H2[0])))
    gamma_0_H2 = np.array(broad_file_H2[1])
    n_L_H2 = np.array(broad_file_H2[2])

    # Read in He broadening file
    broad_file_He = pd.read_csv(input_directory + 'He.broad',
                                sep = ' ', header=None, skiprows=1)
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

    return J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He


def read_air(input_directory):
    '''
    Read the air broadening file from the input directory

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    J_max : TYPE
        DESCRIPTION.
    gamma_0_air : TYPE
        DESCRIPTION.
    n_L_air : TYPE
        DESCRIPTION.

    '''

    # Read in air broadening file
    broad_file_air = pd.read_csv(input_directory + 'air.broad',
                                 sep = ' ', header=None, skiprows = 1)
    J_max = int(np.max(np.array(broad_file_air[0])))
    gamma_0_air = np.array(broad_file_air[1])
    n_L_air = np.array(broad_file_air[2])

    return J_max, gamma_0_air, n_L_air


def read_SB07(input_directory):
    '''
    Read the Burrows broadening file from the input directory

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    J_max : TYPE
        DESCRIPTION.
    gamma_0_SB07 : TYPE
        DESCRIPTION.

    '''

    # Read in Sharp & Burrows (2007) broadening file
    broad_file_SB07 = pd.read_csv(input_directory + 'SB07.broad',
                                           sep = ' ', header=None, skiprows=1)
    J_max = int(np.max(np.array(broad_file_SB07[0])))
    gamma_0_SB07 = np.array(broad_file_SB07[1])
    #n_L_SB07 = np.array(broad_file_SB07[2])       # Not really needed, as temperature exponent = 0 for all J''

    return J_max, gamma_0_SB07


def read_custom(input_directory):
    '''
    Read a user-provided broadening file from the input directory

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    J_max : TYPE
        DESCRIPTION.
    gamma_0_air : TYPE
        DESCRIPTION.
    n_L_air : TYPE
        DESCRIPTION.

    '''

    # Read in custom broadening file
    broad_file_custom = pd.read_csv(input_directory + 'custom.broad',
                                    sep = ' ', header=None, skiprows = 1)
    J_max = int(np.max(np.array(broad_file_custom[0])))
    gamma_0_air = np.array(broad_file_custom[1])
    n_L_air = np.array(broad_file_custom[2])

    return J_max, gamma_0_air, n_L_air


def gamma_L_VALD(gamma_vdw, m_s, broadener):
    '''
    Computes Lorentzian HWHM at 296K and 1 bar for a given broadener from a
    tabulated VALD van der Waals broadening constant.

    Parameters
    ----------
    gamma_vdw : float
        van der Waals parameter from VALD.
    m_s : float
        mass of species whose spectral line is being broadened (u).
    broadener : str
        identity of broadening species (H2 or He).

    Returns
    -------
    gamma_L_0 : float
        Lorentzian HWHM at reference T (296K) and P (1 bar).
    n_L : float
        Temperature exponent (fixed to -0.7 for van der Waals theory).

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
    '''
    Computes Lorentzian HWHM at 296K and 1 bar for a given broadener using
    van der Waals impact theory.

    Parameters
    ----------
    E_low : float
        Lower level energy (cm^-1).
    E_up : float
        Upper level energy (cm^-1).
    l_low : int
        Lower level orbital angular momentum.
    l_up : int
        Upper level orbital angular momentum.
    species : str
        Identity of species whose spectral line is being broadened.
    m_s : float
        Mass of species whose spectral line is being broadened (u).
    broadener : str
        Identity of broadening species (H2 or He).

    Returns
    -------
    gamma_L_0 : float
        Lorentzian HWHM at reference T (296K) and P (1 bar).
    n_L : float
        Temperature exponent (fixed to -0.7 for van der Waals theory).

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
    n_low_sq = ((Ryd/100.0) * (Z + 1.0)**2)/(E_inf - E_low)
    n_up_sq =  ((Ryd/100.0) * (Z + 1.0)**2)/(E_inf - E_up)

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


def read_atom(species, nu_0, gf, E_low, E_up, J_low, l_low, l_up,
              gamma_nat, gamma_vdw, alkali, m):

    if alkali:  # Special treatments for alkali Van der Waals widths

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

    else:  # For non-alkali species

        gamma_0_H2, n_L_H2 = gamma_L_VALD(gamma_vdw, (m/u), 'H2')
        gamma_0_He, n_L_He = gamma_L_VALD(gamma_vdw, (m/u), 'He')

    return gamma_0_H2, n_L_H2, gamma_0_He, n_L_He


def compute_H2_He(gamma_0_H2, T_ref, T, n_L_H2, P, P_ref, X_H2, gamma_0_He, n_L_He, X_He):
    gamma = (gamma_0_H2 * np.power((T_ref/T), n_L_H2) * (P/P_ref) * X_H2 +   # H2+He Lorentzian HWHM for given T, P, and J (ang. mom.)
             gamma_0_He * np.power((T_ref/T), n_L_He) * (P/P_ref) * X_He)    # Note that these are only a function of J''

    return gamma

def compute_air(gamma_0_air, T_ref, T, n_L_air, P, P_ref):
    gamma = (gamma_0_air * np.power((T_ref/T), n_L_air) * (P/P_ref))      # Air-broadened Lorentzian HWHM for given T, P, and J (ang. mom.)

    return gamma

def compute_SB07(gamma_0_SB07, P, P_ref):
    gamma = (gamma_0_SB07 * (P/P_ref))      # Equation (15) in Sharp & Burrows (2007)

    return gamma
