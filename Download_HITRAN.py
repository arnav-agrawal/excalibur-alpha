#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:07:01 2020

@author: arnav
"""

from hapi import partitionSum, db_begin, fetch, abundance, moleculeName, isotopologueName
import os
import numpy
import h5py
import time
import pandas


def create_directories(mol_ID, iso_ID):
    """
    Create new folders to store the relevant data

    Parameters
    ----------
    mol_ID : TYPE
        DESCRIPTION.
    iso_ID : TYPE
        DESCRIPTION.

    Returns
    -------
    line_list_folder : TYPE
        DESCRIPTION.

    """
    
    input_folder = '../input'
    molecule_folder = input_folder + '/' + moleculeName(mol_ID) + '  |  ' + isotopologueName(mol_ID, iso_ID)
    line_list_folder = molecule_folder + '/HITRAN/'
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if os.path.exists(molecule_folder) == False:
        os.mkdir(molecule_folder)

    if os.path.exists(line_list_folder) == False:
        os.mkdir(line_list_folder)
        
    return line_list_folder


def create_pf(mol_ID, iso_ID, folder, T_min = 70, T_max = 3001, step = 1.0):
    """
    Create partition function file using the partitionSum() function already in HITRAN

    Parameters
    ----------
    mol_ID : TYPE
        DESCRIPTION.
    iso_ID : TYPE
        DESCRIPTION.
    folder : TYPE
        DESCRIPTION.
    T_min : TYPE, optional
        DESCRIPTION. The default is 70.
    T_max : TYPE, optional
        DESCRIPTION. The default is 3001.
    step : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    None.

    """
    
    T, Q = partitionSum(mol_ID, iso_ID, [T_min, T_max], step)

    out_file = folder + moleculeName(mol_ID) + '.pf'
    f_out = open(out_file, 'w')
    f_out.write('T | Q \n') 
    for i in range(len(T)):
        f_out.write('%.1f %.4f \n' %(T[i], Q[i]))
        

def download_trans_file(mol_ID, iso_ID, folder, nu_min = 200, nu_max = 25000):
    """
    Download line list using the fetch() function already in HITRAN

    Parameters
    ----------
    mol_ID : TYPE
        DESCRIPTION.
    iso_ID : TYPE
        DESCRIPTION.
    folder : TYPE
        DESCRIPTION.
    nu_min : TYPE, optional
        DESCRIPTION. The default is 200.
    nu_max : TYPE, optional
        DESCRIPTION. The default is 25000.

    Returns
    -------
    None.

    """
    db_begin(folder)
    fetch(moleculeName(mol_ID), mol_ID, iso_ID, nu_min, nu_max)
    
    
   

def convert_to_hdf(mol_ID, iso_ID, file):
    """
    Convert a given file to HDF5 format. Used for the .trans files.

    Parameters
    ----------
    mol_ID : TYPE
        DESCRIPTION.
    iso_ID : TYPE
        DESCRIPTION.
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    J_lower : TYPE
        DESCRIPTION.
    gamma_L_0_air : TYPE
        DESCRIPTION.
    n_L_air : TYPE
        DESCRIPTION.

    """
    
    start_time = time.time()
    
    # Different HITRAN formats for different molecules leads us to read in .par files w/ different field widths
    if mol_ID in {1, 3, 9, 12, 20, 21, 25, 29, 31, 32, 35, 37, 38}:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 3, 5, 1, 6, 12, 1, 7, 7]
        J_col = 13
        
    if mol_ID in {10, 33}:  #Handle HO2 and NO2 J_cols since HITRAN provides N instead of J
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 3, 5, 1, 6, 12, 1, 7, 7]
        J_col = 13
        Sym_col = 17
    
    if mol_ID in {2, 4, 5, 14, 15, 16, 17, 19, 22, 23, 26, 36}:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 5, 1, 3, 1, 5, 6, 12, 1, 7, 7]
        J_col = 15
        
    if (mol_ID == 6 and iso_ID in {1, 2}) or mol_ID == 30:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 2, 3, 2, 3, 5, 6, 12, 1, 7, 7]
        J_col = 14
        
    if mol_ID in {11, 24, 27, 28, 39} or (mol_ID == 6 and iso_ID in {3, 4}):
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 2, 2, 1, 4, 6, 12, 1, 7, 7]
        J_col = 13
        
    if mol_ID == 7:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 1, 1, 3, 1, 3, 5, 1, 6, 12, 1, 7, 7]
        J_col = 17
        
    if mol_ID in {8, 18}: #removed 13 for now
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 1, 5, 1, 5, 6, 12, 1, 7, 7]
        J_col = 15
        
    if mol_ID in {13}:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 1, 5, 1, 5, 6, 12, 1, 7, 7]
        J_col = 15
        
    
    trans_file = pandas.read_fwf(file, widths=field_lengths, header=None)
    
    # Get only the necessary columns from the .par file
    nu_0 = numpy.array(trans_file[2])
    log_S_ref = numpy.log10(numpy.array(trans_file[3]) / abundance(mol_ID, iso_ID))
    gamma_L_0_air = numpy.array(trans_file[5]) / 1.01325   # Convert from cm^-1 / atm -> cm^-1 / bar
    E_lower = numpy.array(trans_file[7])
    n_L_air = numpy.array(trans_file[8])
    J_lower = numpy.array(trans_file[J_col])
    
    if mol_ID in {10, 33}:  # Handle creation of NO2 and HO2 J_lower columns, as the given value is N on HITRAN not J
        Sym = numpy.array(trans_file[Sym_col])
        for i in range(len(J_lower)):
            if Sym[i] == '+':
                J_lower[i] += 1/2
            else:
                J_lower[i] -= 1/2
    
    hdf_file_path = os.path.splitext(file)[0] + '.h5'
    
    # Write the data to our HDF5 file
    with h5py.File(hdf_file_path, 'w') as hdf:
        hdf.create_dataset('Transition Wavenumber', data = nu_0, dtype = 'f4') #store as 32-bit unsigned float
        hdf.create_dataset('Log Line Intensity', data = log_S_ref, dtype = 'f4') 
        hdf.create_dataset('Lower State E', data = E_lower, dtype = 'f4') 
        hdf.create_dataset('Lower State J', data = J_lower, dtype = 'f4') 
        hdf.create_dataset('Air Broadened Width', data = gamma_L_0_air, dtype = 'f4')
        hdf.create_dataset('Temperature Dependence of Air Broadening', data = n_L_air, dtype = 'f4')

    os.remove(file)
    
    print("This file took", round(time.time() - start_time, 1), "seconds to reformat to HDF.")
    

def create_air_broad(input_dir):
    """
    Create an air broadening file using the downloaded line list

    Parameters
    ----------
    input_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Instantiate arrays which will be needed for creating air broadening file
    J_lower_all, gamma_air, n_air = (numpy.array([]) for _ in range(3))
    gamma_air_avg, n_air_avg = (numpy.array([]) for _ in range(2))
    
    for file in os.listdir(input_dir):
        if file.endswith('.h5'):
            with h5py.File(input_dir + file, 'r') as hdf:
                
                # Populate the arrays by reading in each hdf5 file
                J_lower_all = numpy.append(J_lower_all, numpy.array(hdf.get('Lower State J')))
                gamma_air = numpy.append(gamma_air, numpy.array(hdf.get('Air Broadened Width')))
                n_air = numpy.append(n_air, numpy.array(hdf.get('Temperature Dependence of Air Broadening')))
            
    J_sorted = numpy.sort(numpy.unique(J_lower_all))
        
    for i in range(len(J_sorted)):
        
        gamma_air_i = numpy.mean(gamma_air[numpy.where(J_lower_all == J_sorted[i])])
        n_air_i = numpy.mean(n_air[numpy.where(J_lower_all == J_sorted[i])])
        gamma_air_avg = numpy.append(gamma_air_avg, gamma_air_i)
        n_air_avg = numpy.append(n_air_avg, n_air_i)
        
    # Write air broadening file
    out_file = input_dir + 'air.broad'
    f_out = open(out_file,'w')
    
    f_out.write('J | gamma_L_0 | n_L \n')
    
    for i in range(len(J_sorted)):
        f_out.write('%.1f %.4f %.4f \n' %(J_sorted[i], gamma_air_avg[i], n_air_avg[i]))
    
    f_out.close()


    

def summon_HITRAN(molecule, isotopologue):
    """
    Main function, uses calls to other functions to perform the download

    Parameters
    ----------
    molecule : TYPE
        DESCRIPTION.
    isotopologue : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    print("\nFetching data from HITRAN...\nMolecule:", moleculeName(molecule), "\nIsotopologue", isotopologueName(molecule, isotopologue), "\n")
    
    output_folder = create_directories(molecule, isotopologue)
    create_pf(molecule, isotopologue, output_folder)
    download_trans_file(molecule, isotopologue, output_folder)
    for file in os.listdir(output_folder):
        if file.endswith('.data'):
            convert_to_hdf(molecule, isotopologue, output_folder + file)
    create_air_broad(output_folder)

    return output_folder