#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:43:38 2020

@author: arnav
"""

import numpy as np
import pandas as pd
import os
import h5py
import shutil
import re

def process_VALD_file(species, ionization_state):
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
        
    
    directory = '../VALD Line Lists/'

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
    
    convert_to_hdf(directory + species + '_' + roman_ion + '.trans', alkali)
    
    
def create_pf_VALD():
    """
    Used on developers' end to create the partition function file which is included in the GitHub package
    

    Returns
    -------
    None.

    """
    
    fname = './Atom_partition_functions.txt'
    
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
    
    pf.to_csv('./Atomic_partition_functions.pf')

def convert_to_hdf(file, alkali):
    
    trans_file = pd.read_csv(file, delim_whitespace = True, header=None, skiprows = 1)
    
    nu_0 = np.array(trans_file[0])
    log_gf = np.array(trans_file[1])
    E_low = np.array(trans_file[2])
    E_up = np.array(trans_file[3])
    J_low = np.array(trans_file[4])
    J_up = np.array(trans_file[5])
    
    if alkali:  # Account for the differences in columns depending on if the species is an alkali metal
        l_low = np.array(trans_file[6])
        l_up = np.array(trans_file[7])
        log_gamma_nat = np.array(trans_file[8])
        log_gamma_vdw = np.array(trans_file[9])
        
    else:
        log_gamma_nat = np.array(trans_file[6])
        log_gamma_vdw = np.array(trans_file[7])
        
        
    hdf_file_path = os.path.splitext(file)[0] + '.h5'
    
    with h5py.File(hdf_file_path, 'w') as hdf:
        hdf.create_dataset('nu', data = nu_0, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('Log gf', data = log_gf, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('E lower', data = E_low, dtype = 'f8') #store as 32-bit float
        hdf.create_dataset('E upper', data = E_up, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('J lower', data = J_low, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('J upper', data = J_up, dtype = 'f8') #store as 32-bit float
        hdf.create_dataset('Log gamma nat', data = log_gamma_nat, dtype = 'f8') #store as 32-bit float
        hdf.create_dataset('Log gamma vdw', data = log_gamma_vdw, dtype = 'f8') #store as 32-bit float
        
        if alkali:
            hdf.create_dataset('l lower', data = l_low, dtype = 'f8') #store as 32-bit unsigned int
            hdf.create_dataset('l upper', data = l_up, dtype = 'f8') #store as 32-bit float

    # os.remove(file)       
            
def create_directories(molecule, ionization_state):
    roman_num = ''
    for i in range(ionization_state):
        roman_num += 'I'
        
    fname = molecule + '_' + roman_num + '.h5'
    
    input_folder = '../input'
    molecule_folder = input_folder + '/' + molecule + '  ~  (' + roman_num + ')'
    line_list_folder = molecule_folder + '/VALD'
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if os.path.exists(molecule_folder) == False:
        os.mkdir(molecule_folder)

    if os.path.exists(line_list_folder) == False:
        os.mkdir(line_list_folder)
    
    shutil.copy('../VALD Line Lists/' + fname, line_list_folder + '/') # Copy the line list file to the newly created folder
    
    return line_list_folder

  
def filter_pf(molecule, ionization_state, line_list_folder):
    ionization_state_roman = ''
    
    for i in range(ionization_state):
        ionization_state_roman += 'I'
        
    all_pf = pd.read_csv('../VALD Line Lists/Atomic_partition_functions.pf', index_col = 0, )
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
    
          
def summon_VALD(molecule, ionization_state):
    print("\n ***** Processing requested data from VALD. You have chosen the following parameters: ***** ")
    print("\nAtom:", molecule, "\nIonization State:", ionization_state)
    line_list_folder = create_directories(molecule, ionization_state) # In this I will want to move the .h5 file to the right directory
    filter_pf(molecule, ionization_state, line_list_folder)
  
    
