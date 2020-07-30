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

def process_VALD_file(species):
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
    
    directory = '../VALD Line Lists/'

    trans_file = [filename for filename in os.listdir(directory) if filename == (species + '_VALD.trans')]

    wl = []
    log_gf = []
    E_low = []
    E_up = []
    l_low = []
    l_up = []
    J_low = []
    J_up = []
    log_gamma_vdw = []

    f_in = open(directory + trans_file[0], 'r')

    count = 0
    
    debug = 0

    for line in f_in:

        count += 1
        

        if (count >= 3):

            if ((count+1)%4 == 0):

                line = line.strip()
                line = line.split(',')

                # If at beginning of file footnotes, do not read further
                if (line[0] == '* oscillator strengths were scaled by the solar isotopic ratios.'): break
            
                if debug <= 5:
                    print(line)
                    print(line[3])
                    print(line[4])
                    print(line[5])
                    debug+=1
                
                wl.append(float(line[1]))   # Convert wavelengths to um
                log_gf.append(float(line[2]))
                E_low.append(float(line[3]))
                J_low.append(float(line[4]))
                E_up.append(float(line[5]))
                J_up.append(float(line[6]))
                log_gamma_vdw.append(float(line[12]))

            elif ((count)%4 == 0):

                # Only need orbital angular momentum QNs for alkalis
                if (species in ['Li', 'Na', 'K', 'Rb', 'Cs']):

                    line = line.strip()
                    line = line.split()

                    # Orbital angular momentum quntum numbers
                    if   (line[2].endswith('s')): l_low.append(0)
                    elif (line[2].endswith('p')): l_low.append(1)
                    elif (line[2].endswith('d')): l_low.append(2)
                    elif (line[2].endswith('f')): l_low.append(3)
                    elif (line[2].endswith('g')): l_low.append(4)
                    else: print ("Error: above g orbital!")

            elif ((count-1)%4 == 0):

                # Only need orbital angular momentum QNs for alkalis
                if (species in ['Li', 'Na', 'K', 'Rb', 'Cs']):

                    line = line.strip()
                    line = line.split()

                    # Orbital angular momentum quntum numbers
                    if   (line[2].endswith('s')): l_up.append(0)
                    elif (line[2].endswith('p')): l_up.append(1)
                    elif (line[2].endswith('d')): l_up.append(2)
                    elif (line[2].endswith('f')): l_up.append(3)
                    elif (line[2].endswith('g')): l_up.append(4)
                    else: print ("Error: above g orbital!")

    f_in.close()

    nu = 1.0e4/np.array(wl)
    nu = nu[::-1]

    # Reverse array directions for increasing wavenumber
    wl = np.array(wl[::-1]) * 1.0e-3       # Convert nm to um
    log_gf = np.array(log_gf[::-1])
    E_low = np.array(E_low[::-1]) * 8065.547574991239  # Convert eV to cm^-1
    E_up = np.array(E_up[::-1]) * 8065.547574991239
    l_low = np.array(l_low[::-1])
    l_up = np.array(l_up[::-1])
    J_low = np.array(J_low[::-1])
    J_up = np.array(J_up[::-1])
    log_gamma_vdw = np.array(log_gamma_vdw[::-1])
    
    print(len(wl), len(log_gf), len(E_low), len(E_up), len(J_low), len(J_up), len(l_low), len(l_up), len(log_gamma_vdw))

    # Compute transition wavenumbers
    nu = 1.0e4/np.array(wl)

    # Compute gf factor
    gf = np.power(10.0, log_gf)

    # Open output file
    f_out = open(directory + 'K.trans','w')

    f_out.write('nu_0 | gf | E_low | E_up | J_low | J_up | l_low | l_up | log_gamma_vdw \n')

    for i in range(len(nu)):
        f_out.write('%.6f %.6e %.6f %.6f %.1f %.1f %d %d %.6f \n' %(nu[i], gf[i], E_low[i], E_up[i],
                                                                    J_low[i], J_up[i], l_low[i], l_up[i],
                                                                    log_gamma_vdw[i]))
    f_out.close()
    
    convert_to_hdf(directory + 'K.trans')
    
    return


def convert_to_hdf(file):
    
    trans_file = pd.read_csv(file, delim_whitespace = True, header=None, skiprows = 1)
    
    print(trans_file.head())
    
    nu_0 = np.array(trans_file[0])
    log_gf = np.array(trans_file[1])
    E_low = np.array(trans_file[2])
    E_up = np.array(trans_file[3])
    J_low = np.array(trans_file[4])
    J_up = np.array(trans_file[5])
    l_low = np.array(trans_file[6])
    l_up = np.array(trans_file[7])
    log_gamma_vdw = np.array(trans_file[8])
    
    hdf_file_path = os.path.splitext(file)[0] + '.h5'
    
    with h5py.File(hdf_file_path, 'w') as hdf:
        hdf.create_dataset('Nu', data = nu_0, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('Log gf', data = log_gf, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('E lower', data = E_low, dtype = 'f8') #store as 32-bit float
        hdf.create_dataset('E upper', data = E_up, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('J lower', data = J_low, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('J upper', data = J_up, dtype = 'f8') #store as 32-bit float
        hdf.create_dataset('L lower', data = l_low, dtype = 'f8') #store as 32-bit unsigned int
        hdf.create_dataset('L upper', data = l_up, dtype = 'f8') #store as 32-bit float
        hdf.create_dataset('Log gamma vdw', data = log_gamma_vdw, dtype = 'f8') #store as 32-bit float

    # os.remove(file)

# Create directory location and copy .h5 file to that location

process_VALD_file('Fe')
