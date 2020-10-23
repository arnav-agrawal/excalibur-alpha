import os
import numpy as np
import pandas as pd
import re
import time
import requests
import sys
import numba
from bs4 import BeautifulSoup
from scipy.interpolate import UnivariateSpline as Interp
from .hapi import molecularMass, moleculeName, isotopologueName

from .calculate import produce_total_cross_section_VALD_atom, bin_cross_section_atom

from excalibur.constants import c, kb, u, P_ref, T_ref, \
                                gamma_0_fixed, n_L_fixed

import excalibur.ExoMol as ExoMol
import excalibur.HITRAN as HITRAN
import excalibur.HITEMP as HITEMP
import excalibur.VALD as VALD
import excalibur.downloader as download
import excalibur.broadening as broadening
import excalibur.Voigt as Voigt
import excalibur.calculate as calculate

from excalibur.misc import write_output, check_molecule


def mass(species, isotopologue, linelist):
    """
    Determine the mass of a given chemical species-isotopologue combination

    Parameters
    ----------
    species : String
        Molecule we are calculating the mass for.
    isotopologue : String
        Isotopologue of this species we are calculating the mass for.
    linelist : String
        The line list this species' cross-section will later be calculated for. Used to 
        identify between ExoMol, HITRAN/HITEMP, and VALD

    Returns
    -------
    int
        Mass of the given species-isotopologue combination.

    """
    
    # For HITRAN or HITEMP line lists
    if linelist == 'hitran' or linelist == 'hitemp':
        mol_ID = 1
        while moleculeName(mol_ID) != species:
            mol_ID += 1
            
        iso_ID = 1
        
        while True:
            iso_name = isotopologueName(mol_ID, iso_ID) # Need to format the isotopologue name to match ExoMol formatting
    
            # 'H' not followed by lower case letter needs to become '(1H)'
            iso_name = re.sub('H(?![a-z])', '(1H)', iso_name)
    
            # Number of that atom needs to be enclosed by parentheses ... so '(1H)2' becomes '(1H2)'
            matches = re.findall('[)][0-9]{1}', iso_name)
            for match in matches:
                number = re.findall('[0-9]{1}', match)
                iso_name = re.sub('[)][0-9]{1}', number[0] + ')', iso_name)
    
            # replace all ')(' with '-'
            iso_name = iso_name.replace(')(', '-')
            
            if iso_name == isotopologue:
                return molecularMass(mol_ID, iso_ID)
            
            else:
                iso_ID += 1
      
    # For VALD line lists
    elif linelist == 'vald':   
        
        # Atomic masses - Weighted average based on isotopic natural abundances found here: 
        # https://www.chem.ualberta.ca/~massspec/atomic_mass_abund.pdf
        mass_dict = {'H': 1.00794072, 'He': 4.00260165, 'Li': 6.94003706, 'Be': 9.012182, 'B': 10.81102777,
                     'C': 12.0107359, 'N': 14.00674309, 'O': 15.9994053, 'F': 18.998403, 'Ne': 20.1800463,
                     'Na': 22.989770, 'Mg': 24.30505187, 'Al': 26.981538, 'Si': 28.0853852, 'P': 30.973762,
                     'S': 32.06608499, 'Cl': 35.45653261, 'Ar': 39.94767659, 'K': 39.09830144, 
                     'Ca': 40.07802266, 'Sc': 44.955910, 'Ti': 47.86674971, 'Va': 50.941472, 'Cr': 51.99613764,
                     'Mn': 54.938050, 'Fe': 55.84515013, 'Co': 58.933200, 'Ni': 58.69335646, 'Cu': 63.5456439, 
                     'Zn': 65.3955669, 'Ga': 69.72307155, 'Ge': 72.61275896, 'As': 74.921596, 'Se': 78.95938897,
                     'Br': 79.90352862, 'Kr': 83.79932508, 'Rb': 85.46766375, 'Sr': 87.61664598, 
                     'Y': 88.905848, 'Zr': 91.22364739, 'Nb': 92.906378, 'Mo': 95.93129084, 'Ru': 101.06494511,
                     'Rh': 102.905504, 'Pd': 106.41532721, 'Ag': 107.8681507, 'Cd': 112.41155267, 
                     'In': 114.81808585, 'Sn': 118.71011064, 'Sb': 121.7597883, 'Te': 127.60312538, 
                     'I': 126.904468, 'Xe': 131.29248065, 'Cs': 132.905447, 'Ba': 137.32688569, 
                     'La': 138.90544868, 'Ce': 140.11572155, 'Pr': 140.907648, 'Nd': 144.23612698, 
                     'Sm': 149.46629229, 'Eu': 151.96436622, 'Gd': 157.25211925, 'Tb': 158.925343, 
                     'Dy': 162.49703004, 'Ho': 164.930319, 'Er': 167.25630107, 'Tm': 168.934211, 
                     'Yb': 173.0376918, 'Lu': 174.96671757, 'Hf': 178.48497094, 'Ta': 180.94787594, 
                     'W': 183.84177868, 'Re': 186.20670567, 'Os': 190.22755215, 'Ir': 192.21605379, 
                     'Pt': 194.73875746, 'Au': 196.966552, 'Hg': 200.59914936, 'Tl': 204.38490867, 
                     'Pb': 207.21689158, 'Bi': 208.980383, 'Th': 232.038050, 'Pa': 231.035879, 'U': 238.02891307
                     }
        
        return mass_dict.get(species)

    # For ExoMol line lists
    else:
        
        isotopologue = isotopologue.replace('(', '')
        isotopologue = isotopologue.replace(')', '')

        url = 'http://exomol.com/data/molecules/' + species + '/' + isotopologue + '/' + linelist + '/'
        
        # Parse the webpage to find the .def file and read it
        web_content = requests.get(url).text
        soup = BeautifulSoup(web_content, "lxml")
        def_tag = soup.find('a', href = re.compile("def"))
        new_url = 'http://exomol.com' + def_tag.get('href')
        
        out_file = './def'
        with requests.get(new_url, stream=True) as request:
            with open(out_file, 'wb') as file:
                for chunk in request.iter_content(chunk_size = 1024 * 1024):
                    file.write(chunk)
                    
        data = pd.read_csv(out_file, delimiter = '#', names = ['Value', 'Key'])  # Store the .def file in a pandas DataFrame
        data = data[data['Key'].str.contains('mass')]  # Only use the row that contains the isotopologue mass
        data = data.reset_index(drop = True)  # Reset the index of the DataFrame
        mass = data['Value'][0]
        mass = re.findall('[0-9|.]+', mass)[0]
        
        os.remove(out_file)
        
        return float(mass)
        

def load_pf(input_directory):
    '''
    Read in a pre-downloaded partition function.

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    T_pf_raw : TYPE
        DESCRIPTION.
    Q_raw : TYPE
        DESCRIPTION.

    '''
    
    print("Loading partition function")
    
    # Look for files in input directory ending in '.pf'
    pf_file_name = [filename for filename in os.listdir(input_directory) if filename.endswith('.pf')]
    
    # Read partition function
    pf_file = pd.read_csv(input_directory + pf_file_name[0], sep= ' ', header=None, skiprows=1)
    
    # First column in standard format is temperature, second is the partition function
    T_pf_raw = np.array(pf_file[0]).astype(np.float64)
    Q_raw = np.array(pf_file[1])
    
    return T_pf_raw, Q_raw


def interpolate_pf(T_pf_raw, Q_raw, T, T_ref):
    '''
    Interpolate partition function to the temperature of the cross section computation.

    Parameters
    ----------
    T_pf_raw : TYPE
        DESCRIPTION.
    Q_raw : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    T_ref : TYPE
        DESCRIPTION.

    Returns
    -------
    Q_T : TYPE
        DESCRIPTION.
    Q_T_ref : TYPE
        DESCRIPTION.

    '''
    
    # Interpolate partition function onto a fine grid using a 5th order spline
    pf_spline = Interp(T_pf_raw, Q_raw, k=5)
    
    # Define a new temperature grid (extrapolated to 10,000K)
    T_pf_fine = np.linspace(1.0, 10000.0, 9999)      
    
    # Using spline, interpolate and extrapolate the partition function to the new T grid
    Q_fine = pf_spline(T_pf_fine)                    
    
    # Find the indices in the fine temperature grid closest to the user specified and reference temperatures
    idx_T = np.argmin(np.abs(T_pf_fine - T))
    idx_T_ref = np.argmin(np.abs(T_pf_fine - T_ref))   
    
    # Find partition function at the user specified and reference temperatures
    Q_T = Q_fine[idx_T]                               
    Q_T_ref = Q_fine[idx_T_ref]                       
    
    return Q_T, Q_T_ref


def create_nu_grid_atom(atom, T, m, gamma, nu_0, Voigt_sub_spacing, 
                        dnu_out, nu_out_min, nu_out_max, Voigt_cutoff, cut_max):
    '''
    Create the computational (fine) and output (coarse) wavenumber grids for
    an atomic cross section calculation.
    
    Note: for atoms a single grid is used over the entire wavenumber range.

    '''
    
    # Define the minimum and maximum wavenumber on grid to go slightly beyond user's output limits
    nu_min = 1
    nu_max = nu_out_max + 1000
    
    # First, we need to find values of gamma_V for reference wavenumber (1000 cm^-1)
    alpha_ref = np.sqrt(2.0*kb*T*np.log(2)/m) * (np.array(1000.0)/c)  # Doppler HWHM at reference wavenumber
    gamma_ref = np.min(gamma)                                            # Find minimum value of Lorentzian HWHM
    gamma_V_ref = Voigt.HWHM(gamma_ref, alpha_ref)                       # Reference Voigt width
    
    # Calculate Voigt width for each transition
    alpha = np.sqrt(2.0*kb*T*np.log(2)/m) * (np.array(nu_0)/c)   # Doppler HWHM for each transition
    gamma_V = Voigt.HWHM(gamma, alpha)   # Voigt HWHM
    
    #**** Now compute properties of computational (fine) and output (coarse) wavenumber grid *****
    
    # Wavenumber spacing of of computational grid (smallest of gamma_V_ref/6 or 0.01cm^-1)
    dnu_fine = np.minimum(gamma_V_ref*Voigt_sub_spacing, dnu_out)      
    
    # Number of points on fine grid (rounded)
    N_points_fine = int((nu_max-nu_min)/dnu_fine + 1)
    
    # Adjust dnu_fine slightly to match an exact integer number of grid spaces
    dnu_fine = (nu_max-nu_min)/(N_points_fine - 1)
    
    cutoffs = np.zeros(len(nu_0))   # Line wing cutoffs for each line
    
    # Line cutoffs at min(500 gamma_V, 1000cm^-1)
    for i in range(len(nu_0)):
        
        cutoffs[i] = dnu_fine * (int((Voigt_cutoff*gamma_V[i])/dnu_fine))

        if (cutoffs[i] >= cut_max): cutoffs[i] = cut_max
                
        # Special cases for alkali resonant lines
        if ((atom == 'Na') and (int(nu_0[i]) in [16978, 16960])):
            cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        elif ((atom == 'K') and  (int(nu_0[i]) in [13046, 12988])): 
            cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
      
    # Calculate detuning frequencies for Na and K resonance lines (after Baudino+2015)
    if (atom == 'Na'): 
        nu_detune = 30.0 * np.power((T/500.0), 0.6)
    elif (atom == 'K'): 
        nu_detune = 20.0 * np.power((T/500.0), 0.6)
    else: 
        nu_detune = cut_max
    
    # Evaluate number of frequency points for each Voigt function up to cutoff (one tail)
    N_Voigt_points = ((cutoffs/dnu_fine).astype(np.int64)) + 1  
    
    # Define start and end points of fine grid
    nu_fine_start = nu_min
    nu_fine_end = nu_max
    
    # Initialise output grid
    N_points_out = int((nu_out_max-nu_out_min)/dnu_out + 1)     # Number of points on coarse grid (uniform)
    nu_out = np.linspace(nu_out_min, nu_out_max, N_points_out)  # Create coarse (output) grid
    
    # Initialise cross section arrays on each grid
    sigma_fine = np.zeros(N_points_fine)    # Computational (fine) grid
    sigma_out = np.zeros(N_points_out)      # Coarse (output) grid
    
    return (sigma_fine, sigma_out, nu_detune, N_points_fine, N_Voigt_points, alpha, 
            cutoffs, nu_min, nu_max, nu_fine_start, nu_fine_end, nu_out, N_points_out)



def create_nu_grid_molecule(nu_out_min, nu_out_max, dnu_out):

    # Define the minimum and maximum wavenumber on grid to go slightly beyond user's output limits
    nu_min = 1
    nu_max = nu_out_max + 1000
        
    # Initialise computational grid
    N_compute = int((nu_max - nu_min)/dnu_out + 1)       # Number of points on computational grid (uniform)
    nu_compute = np.linspace(nu_min, nu_max, N_compute)  # Create coarse (output) grid
      
    return nu_compute
    

def summon(database = '', species = '', isotope = 'default', VALD_data_dir = '',
           linelist = 'default', ionization_state = 1, **kwargs):
    '''
    Makes calls to other downloader files to retrieve the data from the desired database


    Parameters
    ----------
    database : TYPE, optional
        DESCRIPTION. The default is ''.
    species : TYPE, optional
        DESCRIPTION. The default is ''.
    isotope : TYPE, optional
        DESCRIPTION. The default is 'default'.
    linelist : TYPE, optional
        DESCRIPTION. The default is 'default'.
    ionization_state : TYPE, optional
        DESCRIPTION. The default is 1.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # Check if the user has specified a chemical species and line list database
    if database != '' and species != '': 
        user_prompt = False
    else: 
        user_prompt = True
        
    # If the user wants to be guided via terminal prompts
    if user_prompt: 
        
        while True:
            database = input('Which line list database do you wish to download from (ExoMol, HITRAN, HITEMP, or VALD)?\n')
            database = database.lower()
            if database == 'exomol' or database == 'hitran' or database == 'hitemp' or database == 'vald' :
                break
            else:
                print("\n ----- This is not a supported database, please try again ----- ")
        
        if database == 'exomol': 
            mol, iso, lin, URL = ExoMol.determine_linelist()
            ExoMol.summon_ExoMol(mol, iso, lin, URL)
            
        if database == 'hitran':
            mol, iso = HITRAN.determine_linelist()
            HITRAN.summon_HITRAN(mol, iso)
            
        if database == 'hitemp':
            mol, iso = HITEMP.determine_linelist()
            HITEMP.summon_HITEMP(mol, iso)
            
        if database == 'vald':
            mol, ion = VALD.determine_linelist(VALD_data_dir)
            VALD.summon_VALD(mol, ion, VALD_data_dir)
            
    # If the user calls summon with parameters directly passed in
    if not user_prompt: 
        
        db = database.lower()
        spe = species
        
        if isinstance(isotope, str):
            try:
                isotope = int(isotope)
            except ValueError:
                pass
            
        iso = isotope
        lin = linelist
        ion = ionization_state
        
        if db == 'exomol':
            
            spe = re.sub('[+]', '_p', spe)  # Handle ions
            iso = re.sub('[+]', '_p', iso)  # Handle ions
            
            if isotope == 'default':
                ExoMol.check(spe)
                iso = ExoMol.get_default_iso(spe)
            if linelist == 'default':
                ExoMol.check(spe, iso)
                lin = ExoMol.get_default_linelist(spe, iso)

            ExoMol.check(spe, iso, lin)
            URL = "http://exomol.com/data/molecules/" + spe + '/' + iso + '/' + lin + '/'
            ExoMol.summon_ExoMol(spe, iso, lin, URL)
            
        elif db == 'hitran':
            
            if isotope == 'default':
                iso = 1
            
            spe = HITRAN.check(spe, iso)
            HITRAN.summon_HITRAN(spe, iso)
            
        elif db == 'hitemp':
            
            if isotope == 'default':
                iso = 1
                
            spe = HITEMP.check(spe, iso)
            HITEMP.summon_HITEMP(spe, iso)
            
        elif db == 'vald':
            
            VALD.check(spe, ion, VALD_data_dir)
            VALD.summon_VALD(spe, ion, VALD_data_dir)
        
        else:
            print("\n ----- You have not passed in a valid database. Please try calling the summon() function again. ----- ")
            sys.exit(0)
        
    print("\nLine list ready.\n")
    
    
def compute_cross_section(input_dir, database, species, log_pressure, temperature, isotope = 'default', 
                          ionization_state = 1, linelist = 'default', cluster_run = False, 
                          nu_out_min = 200, nu_out_max = 25000, dnu_out = 0.01, broad_type = 'default', 
                          X_H2 = 0.85, X_He = 0.15, Voigt_cutoff = 500, Voigt_sub_spacing = (1.0/6.0), 
                          N_alpha_samples = 500, S_cut = 1.0e-100, cut_max = 30.0, N_cores = 8, **kwargs):
    '''
    Main function to calculate molecular and atomic cross sections.

    '''
    
    print("Beginning cross-section computations...")
    
    # Start clock for timing program
    t_start = time.perf_counter()
    
    # Configure numba to paralelise with the user specified number of cores
    numba.set_num_threads(N_cores)
    
    # Cast log_pressure and temperature to lists if they are not already
    if not isinstance(log_pressure, list) and not isinstance(log_pressure, np.ndarray):  
        log_pressure = [log_pressure]
    
    if not isinstance(temperature, list) and not isinstance(temperature, np.ndarray):  
        temperature = [temperature]
        
    # Cast all temperatures and pressures to floats
    for i in range(len(log_pressure) - 1):
        log_pressure[i] = float(log_pressure[i])
    
    for i in range(len(temperature) - 1):
        temperature[i] = float(temperature[i])
    
    database = database.lower()
    
    # Locate the input_directory where the line list is stored
    input_directory = download.find_input_dir(input_dir, database, species, isotope, ionization_state, linelist)
    
    # Use the input directory to define these right at the start
    linelist, isotopologue = download.parse_directory(input_directory, database)
    
    # HITRAN, HITEMP, and VALD do not have seperate line list names
    if database != 'exomol':
        linelist = database
    
    # Load full set of downloaded line list files
    linelist_files = [filename for filename in os.listdir(input_directory) if filename.endswith('.h5')]
    
    if database == 'exomol':
        print("Loading ExoMol format")
        E, g, J = ExoMol.load_states(input_directory)  # Load from .states file
    
    elif database == 'hitran':
        print("Loading HITRAN format")
        # Nothing else required at this stage
    
    elif database == 'hitemp':
        print("Loading HITEMP format")
        # Nothing else required at this stage
        
    elif database == 'vald':
        print("Loading VALD format")
        nu_0, gf, E_low, E_up, J_low, l_low, l_up, \
        Gamma_nat, Gamma_vdw, alkali = VALD.load_line_list(input_directory, species)
        
    # Load partition function
    T_pf_raw, Q_raw = load_pf(input_directory)
    
    # Find mass of the species
    m = mass(species, isotopologue, linelist) * u

    # Check if we have a molecule or an atom
    is_molecule = check_molecule(species)
    
    # If user didn't specify a type of pressure broadening, determine based on available broadening data
    if is_molecule and broad_type == 'default':
        
        broad_type = broadening.det_broad(input_directory)
        
        if broad_type == 'H2-He':
            J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He = broadening.read_H2_He(input_directory)
            
        elif broad_type == 'air':
            J_max, gamma_0_air, n_L_air = broadening.read_air(input_directory)
            
        elif broad_type == 'SB07':
            J_max, gamma_0_SB07 = broadening.read_SB07(input_directory)
            
    # If user specifed a pressure broadening prescription, proceed to load the relevant broadening file
    elif is_molecule and broad_type != 'default':
        
        if (broad_type == 'H2-He' and 'H2.broad' in os.listdir(input_directory) 
                                  and 'He.broad' in os.listdir(input_directory)):
            J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He = broadening.read_H2_He(input_directory)
        
        elif broad_type == 'air' and 'air.broad' in os.listdir(input_directory):
            J_max, gamma_0_air, n_L_air = broadening.read_air(input_directory)
            
        elif broad_type == 'SB07':
            broadening.create_SB07(input_directory)
            J_max, gamma_0_SB07 = broadening.read_SB07(input_directory)
            
        elif broad_type == 'custom' and 'custom.broad' in os.listdir(input_directory):
            J_max, gamma_0_air, n_L_air = broadening.read_custom(input_directory)
        
        elif broad_type == 'fixed':
            J_max = 0
        
        else:
            print("\nYou did not enter a valid type of pressure broadening. Please try again.")
            sys.exit(0)
            
    # For atoms, only H2-He pressure broadening is currently supported
    elif is_molecule == False:
        
        if broad_type != 'default' and broad_type != 'H2-He':
            print("You did not specify a valid choice of pressure broadening.\n" 
                  "For atoms the only supported option is 'H2-He', so we will continue by using that." )
        
        broad_type = 'H2-He'
        
        gamma_0_H2, gamma_0_He, \
        n_L_H2, n_L_He = broadening.read_atom(species, nu_0, gf, E_low, E_up, 
                                              J_low, l_low, l_up, Gamma_nat, 
                                              Gamma_vdw, alkali, m)
    
    #***** Load pressure and temperature for this calculation *****#
    P_arr = np.power(10.0, log_pressure)  # Pressure array (bar)
    log_P_arr = np.array(log_pressure)    # log_10 (Pressure/bar) array
    T_arr = np.array(temperature)         # Temperature array (K)
    
    # If conducting a batch run on a cluster
    if (cluster_run == True):
            
        try:
            idx_PT = int(sys.argv[1])
            
        except IndexError:
            print("\n----- You need to enter a command line argument if cluster_run is set to True. ----- ")
            sys.exit(0)
            
        except ValueError:
            print("\n----- The command line argument needs to be an int. -----")
            sys.exit(0)
            
        if idx_PT >= len(log_P_arr) * len(T_arr):
            print("\n----- You have provided a command line argument that is out of range for the specified pressure and temperature arrays. -----")
            sys.exit(0)
        
        P = P_arr[idx_PT//len(T_arr)]   # Atmospheric pressure (bar)
        T = T_arr[idx_PT%len(T_arr)]    # Atmospheric temperature (K)
        
        # For a cluster run, each core separately handles a single (P,T) combination
        N_P = 1
        N_T = 1
        
    # If running on a single machine, compute a cross section for each (P,T) pair sequentially
    else:
        
        N_P = len(log_P_arr)
        N_T = len(T_arr)
    
    # Compute cross section for each pressure and temperature point
    for p in range(N_P):
        for t in range(N_T):
            
            # When not running on a cluster, select the next (P,T) pair
            if (cluster_run == False):
                
                P = P_arr[p]   # Atmospheric pressure (bar)
                T = T_arr[t]   # Atmospheric temperature (K)
            
            # Interpolate the tabulated partition function to the desired temperature and reference temperature
            Q_T, Q_T_ref = interpolate_pf(T_pf_raw, Q_raw, T, T_ref)
            
            # Handle pressure broadening, wavenumber grid creation and Voigt profile pre-computation for molecules
            if is_molecule:
                
                # Compute Lorentzian HWHM as a function of J_low
                if broad_type == 'H2-He':
                    gamma = broadening.compute_H2_He(gamma_0_H2, T_ref, T, 
                                                     n_L_H2, P, P_ref, X_H2, 
                                                     gamma_0_He, n_L_He, X_He)
                elif broad_type == 'air':
                    gamma = broadening.compute_air(gamma_0_air, T_ref, T, 
                                                   n_L_air, P, P_ref)
                elif broad_type == 'SB07':
                    gamma = broadening.compute_SB07(gamma_0_SB07, P, P_ref)
                elif broad_type == 'custom': 
                    gamma = broadening.compute_air(gamma_0_air, T_ref, T,    # Computation step is the same as for air broadening
                                                   n_L_air, P, P_ref)
                elif broad_type == 'fixed':
                    gamma = np.array([(gamma_0_fixed * np.power((T_ref/T), n_L_fixed) * (P/P_ref))])  # Fixed Lorentizian HWHM (1 element array)
                    
                # Create wavenumber grid for cross section compuation
                nu_compute = create_nu_grid_molecule(nu_out_min, nu_out_max, dnu_out)
                                                                                                                            
                # Initialise cross section arrays for computations
                sigma_compute = np.zeros(len(nu_compute))    # Computational grid
                
                #***** Pre-compute Voigt function array for molecules *****#
    
                print('Pre-computing Voigt profiles...')
    
                t1 = time.perf_counter()    
                
                # Pre-compute template Voigt profiles
                (nu_sampled, alpha_sampled, 
                 cutoffs, N_Voigt, Voigt_arr, 
                 dV_da_arr, dV_dnu_arr, dnu_Voigt) = Voigt.precompute(nu_compute, dnu_out, m, T, 
                                                                      Voigt_sub_spacing, Voigt_cutoff, 
                                                                      N_alpha_samples, gamma, cut_max)
                
                t2 = time.perf_counter()
                time_precompute = t2-t1
            
                print('Voigt profiles computed in ' + str(time_precompute) + ' s')  
                
            # Handle pressure broadening and wavenumber grid creation for atoms
            elif is_molecule == False:  
                
                # Compute Lorentzian HWHM line-by-line for atoms
                gamma = broadening.compute_H2_He(gamma_0_H2, T_ref, T, n_L_H2, 
                                                 P, P_ref, X_H2, gamma_0_He, 
                                                 n_L_He, X_He)
                
                # Add natural broadening for each line
                gamma += ((1.0/(4.0*np.pi*(100.0*c))) * Gamma_nat)  
            
                # Create wavenumber grid properties for cross section calculation
                (sigma_fine, sigma_out, 
                 nu_detune, N_points_fine, 
                 N_Voigt_points, alpha, 
                 cutoffs, nu_min, nu_max, 
                 nu_fine_start, nu_fine_end, 
                 nu_out, N_points_out) = create_nu_grid_atom(species, T, m, gamma, nu_0, 
                                                             Voigt_sub_spacing, dnu_out, 
                                                             nu_out_min, nu_out_max, 
                                                             Voigt_cutoff, cut_max)
                                                                                                                                        
            print("Pre-computation steps complete")
            
            print('Generating cross section for ' + species + ' at P = ' + str(P) + ' bar, T = ' + str(T) + ' K')
            
            # Call relevant cross section computation function for given line list
            if database == 'exomol':    
                calculate.cross_section_EXOMOL(linelist_files, input_directory, 
                                               nu_compute, sigma_compute, alpha_sampled, 
                                               m, T, Q_T, g, E, J, J_max, N_Voigt, cutoffs,
                                               Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt, S_cut)
                
            elif database in ['hitran', 'hitemp']:
                calculate.cross_section_HITRAN(linelist_files, input_directory, 
                                               nu_compute, sigma_compute, alpha_sampled, 
                                               m, T, Q_T, Q_T_ref, J_max, N_Voigt, cutoffs,
                                               Voigt_arr, dV_da_arr, dV_dnu_arr, dnu_Voigt, S_cut)
                
            elif database == 'vald':
                produce_total_cross_section_VALD_atom(sigma_fine, nu_0, nu_detune, E_low, gf, m, T, Q_T,
                                                      N_points_fine, N_Voigt_points, alpha, gamma, cutoffs,
                                                      nu_min, nu_max, S_cut, species)
                    
            if is_molecule:
                
                # Clip ends from computational grid to leave output wavenumber and cross section grids            
                nu_out = nu_compute[(nu_compute >= nu_out_min) & (nu_compute <= nu_out_max)]
                sigma_out = sigma_compute[(nu_compute >= nu_out_min) & (nu_compute <= nu_out_max)]
        
            else:
                bin_cross_section_atom(sigma_fine, sigma_out, nu_fine_start, 
                                       nu_fine_end, nu_out, N_points_fine, N_points_out, 0)
            
            # Create output directory (if not already present)
            output_directory = re.sub('/input/', '/output/', input_directory)
    
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
    
            # Write cross section to file
            write_output(output_directory, species, T, np.log10(P), nu_out, sigma_out)
    
    # Print final runtime
    t_final = time.perf_counter()
    total_final = t_final-t_start
    
    print('Total runtime: ' + str(total_final) + ' s')
    
    return nu_out, sigma_out
