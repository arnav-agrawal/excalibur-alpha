import sys
import os
import numpy as np
import pandas as pd
import h5py

import excalibur.downloader as download


def check(mol, ion, VALD_data_dir):
    '''
    

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.
    ion : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    print(VALD_data_dir)
    roman_num = ''
    for i in range(ion):
            roman_num += 'I'
    fname = mol + '_' + roman_num + '.h5'
    if fname not in os.listdir(VALD_data_dir): 
        print("\n ----- The VALD line list for this atom/isotope combination does not exist. Please try again. -----")
        sys.exit(0)
        
        
def determine_linelist(VALD_data_dir):
    
    while True:
        molecule = input("What atom would you like to download the line list for? \n")
        fname = molecule + '_I.h5'  # Check if at least the neutral version of this atom is supported (i.e. that we even provide the line list for this atom)
        if fname in os.listdir(VALD_data_dir): 
            break
        else:
            print("\n ----- The VALD line list for this atom does not exist. Please try again. -----")
    
    while True:
        try:
            ionization_state = int(input("What ionization state of this atom would you like to download the line list for? ('1' is the neutral state) \n"))
        
        except ValueError:
             print("\n ----- Please enter an integer for the ionization state -----")
        
        else:
            roman_num = ''
            for i in range(ionization_state):
                roman_num += 'I'
        
            fname = molecule + '_' + roman_num + '.h5'  # Check if at least the neutral version of this atom is supported (i.e. that we even provide the line list for this atom)
            if fname in os.listdir(VALD_data_dir): 
                return molecule, ionization_state
            else:
                print("\n ----- The VALD line list for this atom/ionization state combination does not exist. Please try again. -----")


def create_pf_VALD(VALD_data_dir):
    """
    Used on developers' end to create the partition function file which is included in the GitHub package
    

    Returns
    -------
    None.

    """
    
    fname = VALD_data_dir + 'Atom_partition_functions.txt'
    
    temperature = [1.00000e-05, 1.00000e-04, 1.00000e-03, 1.00000e-02, 1.00000e-01, 
                   1.50000e-01, 2.00000e-01, 3.00000e-01, 5.00000e-01, 7.00000e-01, 
                   1.00000e+00, 1.30000e+00, 1.70000e+00, 2.00000e+00, 3.00000e+00, 
                   5.00000e+00, 7.00000e+00, 1.00000e+01, 1.50000e+01, 2.00000e+01, 
                   3.00000e+01, 5.00000e+01, 7.00000e+01, 1.00000e+02, 1.30000e+02, 
                   1.70000e+02, 2.00000e+02, 2.50000e+02, 3.00000e+02, 5.00000e+02, 
                   7.00000e+02, 1.00000e+03, 1.50000e+03, 2.00000e+03, 3.00000e+03, 
                   4.00000e+03, 5.00000e+03, 6.00000e+03, 7.00000e+03, 8.00000e+03, 
                   9.00000e+03, 1.00000e+04]
    
    pf = pd.read_csv(fname, sep = '|', header = 7, skiprows = [0-6, 8, 293], names = temperature)
    
    pf.to_csv(VALD_data_dir + 'Atomic_partition_functions.pf')
    
    
def filter_pf(molecule, ionization_state, line_list_folder, VALD_data_dir):
    ionization_state_roman = ''
    
    for i in range(ionization_state):
        ionization_state_roman += 'I'
        
    all_pf = pd.read_csv(VALD_data_dir + 'Atomic_partition_functions.pf', index_col = 0, )
    all_pf = all_pf.rename(lambda x: x.strip())  # Remove the extra white space in the index names, eg: '  H_I' becomes 'H_I'
    
    pf = all_pf.loc[molecule + '_' + ionization_state_roman]  # Filter the partition functions by the specified atom and ionization state
    pf = pf.reset_index()
    pf.columns = ['T', 'Q'] # Rename the columns
    
    fname = molecule + '_' + ionization_state_roman + '.pf'
    
    T_pf = pf['T'].to_numpy()
    T_pf = T_pf.astype(float)
    Q = pf['Q'].to_numpy()
    
    out_file = line_list_folder + '/' + fname
    f_out = open(out_file, 'w')

    f_out.write('T | Q \n') 

    for i in range(len(T_pf)):
        if T_pf[i] < 10.0:
            continue
        f_out.write('%.1f %.4f \n' %(T_pf[i], Q[i]))

    f_out.close()
    
    
def process_VALD_file(species, ionization_state, VALD_data_dir):
    """
    Used on developers' end to get the necessary data from a VALD line list 

    Parameters
    ----------
    species : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    roman_ion = ''
    
    for i in range(ionization_state):
        roman_ion += 'I'
        
    directory = VALD_data_dir

    trans_file = [filename for filename in os.listdir(directory) if filename == (species + '_' + roman_ion + '_VALD.trans')]

    wl = []
    log_gf = []
    E_low = []
    E_up = []
    l_low = []
    l_up = []
    J_low = []
    J_up = []
    log_gamma_nat = []
    log_gamma_vdw = []
    
    alkali = False

    f_in = open(directory + trans_file[0], 'r')

    count = 0

    for line in f_in:

        count += 1
        
        if (count >= 3):

            if ((count+1)%4 == 0):

                line = line.strip()
                line = line.split(',')

                # If at beginning of file footnotes, do not read further
                if (line[0] == '* oscillator strengths were scaled by the solar isotopic ratios.'): break
                if ('BIBTEX ERROR' in line[0]): break
                
                wl.append(float(line[1]))   # Convert wavelengths to um
                log_gf.append(float(line[2]))
                E_low.append(float(line[3]))
                J_low.append(float(line[4]))
                E_up.append(float(line[5]))
                J_up.append(float(line[6]))
                log_gamma_nat.append(float(line[10]))
                log_gamma_vdw.append(float(line[12]))

            elif ((count)%4 == 0):

                # Only need orbital angular momentum QNs for alkalis
                if (species in ['Li', 'Na', 'K', 'Rb', 'Cs']):
                    
                    line = line.strip()
                    line = line.split()
                    
                    lowercase_letters = [c for c in line[2] if c.islower()]
                    last_lower = lowercase_letters[len(lowercase_letters) - 1]
                    
                    # Orbital angular momentum quntum numbers
                    if last_lower == 's': l_low.append(0)
                    elif last_lower == 'p': l_low.append(1)
                    elif last_lower == 'd': l_low.append(2)
                    elif last_lower == 'f': l_low.append(3)
                    elif last_lower == 'g': l_low.append(4)
                    else: print ("Error: above g orbital!")                
                    
                    alkali = True

            elif ((count-1)%4 == 0):

                # Only need orbital angular momentum QNs for alkalis
                if (species in ['Li', 'Na', 'K', 'Rb', 'Cs']):

                    line = line.strip()
                    line = line.split()
                    
                    lowercase_letters = [c for c in line[2] if c.islower()]
                    last_lower = lowercase_letters[len(lowercase_letters) - 1]

                    # Orbital angular momentum quntum numbers
                    if last_lower == 's': l_up.append(0)
                    elif last_lower == 'p': l_up.append(1)
                    elif last_lower == 'd': l_up.append(2)
                    elif last_lower == 'f': l_up.append(3)
                    elif last_lower == 'g': l_up.append(4)
                    else: print ("Error: above g orbital!")

    f_in.close()

    # Reverse array directions for increasing wavenumber
    wl = np.array(wl[::-1]) * 1.0e-4       # Convert angstrom to um
    log_gf = np.array(log_gf[::-1])
    E_low = np.array(E_low[::-1]) * 8065.547574991239  # Convert eV to cm^-1
    E_up = np.array(E_up[::-1]) * 8065.547574991239
    l_low = np.array(l_low[::-1])
    l_up = np.array(l_up[::-1])
    J_low = np.array(J_low[::-1])
    J_up = np.array(J_up[::-1])
    log_gamma_nat = np.array(log_gamma_nat[::-1])
    log_gamma_vdw = np.array(log_gamma_vdw[::-1])

    # Compute transition wavenumbers
    nu = 1.0e4/np.array(wl)

    # Open output file
    f_out = open(directory + species + '_' + roman_ion + '.trans','w')
    
    if alkali:
        f_out.write('nu_0 | gf | E_low | E_up | J_low | J_up | l_low | l_up | log_gamma_nat | log_gamma_vdw \n')
    
    else:
        f_out.write('nu_0 | gf | E_low | E_up | J_low | J_up | log_gamma_nat | log_gamma_vdw \n')

    for i in range(len(nu)):
        
        if alkali:
            f_out.write('%.6f %.6f %.6f %.6f %.1f %.1f %d %d %.6f %.6f \n' %(nu[i], log_gf[i], E_low[i], E_up[i],
                                                                        J_low[i], J_up[i], l_low[i], l_up[i],
                                                                        log_gamma_nat[i], log_gamma_vdw[i]))
            
        else:
            f_out.write('%.6f %.6f %.6f %.6f %.1f %.1f %.6f %.6f \n' %(nu[i], log_gf[i], E_low[i], E_up[i],
                                                                  J_low[i], J_up[i], log_gamma_nat[i], 
                                                                  log_gamma_vdw[i]))
    f_out.close()
    
    download.convert_to_hdf(file = (directory + species + '_' + roman_ion + '.trans'), alkali = alkali)
    
    
def summon_VALD(molecule, ionization_state, VALD_data_dir):
    
    print("\n ***** Processing requested data from VALD. You have chosen the following parameters: ***** ")
    print("\nAtom:", molecule, "\nIonization State:", ionization_state)
    line_list_folder = download.create_directories(molecule = molecule, ionization_state = ionization_state,
                                                   database = 'VALD', VALD_data_dir = VALD_data_dir) 
    filter_pf(molecule, ionization_state, line_list_folder, VALD_data_dir)
  
    
def load_line_list(input_directory, molecule):
    
    fname = [file for file in os.listdir(input_directory) if file.endswith('.h5')][0]  # The directory should only have one .h5 file containing the line list
    
    with h5py.File(input_directory + fname, 'r') as hdf:
        nu_0 = np.array(hdf.get('nu'))
        gf = np.power(10.0, np.array(hdf.get('Log gf')))
        E_low = np.array(hdf.get('E lower'))
        E_up = np.array(hdf.get('E upper'))
        J_low = np.array(hdf.get('J lower'))
        Gamma_nat = np.power(10.0, np.array(hdf.get('Log gamma nat')))
        Gamma_vdw = np.power(10.0, np.array(hdf.get('Log gamma vdw')))
        
        # VALD stores log_gamma = 0.0 where there is no data. Since 10^(0.0) = 1.0, zero these entries
        Gamma_nat[Gamma_nat == 1.0] = 0.0
        Gamma_vdw[Gamma_vdw == 1.0] = 0.0
        
        if molecule in ['Li', 'Na', 'K', 'Rb', 'Cs']:
            alkali = True
            l_low = np.array(hdf.get('l lower'))
            l_up = np.array(hdf.get('l upper'))
            
        else:
            alkali = False
            l_low = []
            l_up = []
        
    # If transitions are not in increasing wavenumber order, rearrange
    order = np.argsort(nu_0)  # Indices of nu_0 in increasing order
    nu_0 = nu_0[order]
    gf = gf[order]
    E_low = E_low[order]
    E_up = E_up[order]
    J_low = J_low[order]
    Gamma_nat = Gamma_nat[order]
    Gamma_vdw = Gamma_vdw[order]
    
    if alkali:
        l_low = l_low[order]
        l_up = l_up[order]
    
    return nu_0, gf, E_low, E_up, J_low, l_low, l_up, Gamma_nat, Gamma_vdw, alkali
        
    return
    