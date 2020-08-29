import sys
import os
import numpy as np
import h5py
from .hapi import partitionSum, moleculeName, isotopologueName

import excalibur.downloader as download


def create_id_dict():
    mol_ID = []
    mol_name = []
    for i in range(1, 50):
        if i == 35: # Skip 35 since moleculeName(35) throws an error from hapi.py
            continue
        else:
            mol_ID.append(i)
            mol_name.append(moleculeName(i))
        
        mol_ID.append(35)
        mol_name.append('ClONO2')

    names_and_IDs = zip(mol_name, mol_ID)

    molecule_dict = dict(names_and_IDs) 
    
    return molecule_dict

 
def check(molecule, isotope):
    """
    Checks if the parameters passed into the summon() function by the user are valid to use with the HITRAN database

    Parameters
    ----------
    molecule : TYPE
        DESCRIPTION.
    isotope : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    molecule_dict = create_id_dict()
    molecule_id_number = molecule_dict.get(molecule)
    
    try:
        partitionSum(molecule_id_number, isotope, [70, 80], step = 1.0)
    except KeyError:
        print("\n ----- This molecule/isotopologue ID combination is not valid in HITRAN. Please try calling the summon() function again. Make sure you enter the ID number of the molecule you want. ----- ")
        print("\n ----- A list of supported molecule IDs can be found here: https://hitran.org/lbl/ -----")
        sys.exit(0)
        
    return molecule_id_number


def determine_linelist():
    """
    Determines the molecule and isotopologue that would like to be accessed based on user input

    Returns
    -------
    molecule_ID : int
        Molecular ID number of the desired molecule (based on HITRAN conventions found at https://hitran.org/lbl/# ).
    isotopologue_ID : int
        Isotopologue ID number of the desired molecule. 1 is the most abundant isotopologue of that molecule, 2 the next most abundant, and so on.

    """
    
    molecule_dict = create_id_dict()
    
    while True:
        try:
            molecule = input("What molecule would you like to download the line list for? \n")
            molecule_id_number = molecule_dict.get(molecule)
            partitionSum(molecule_id_number, 1, [70, 80], step = 1.0)
        except KeyError:
            print("\n ----- This molecule is not valid. Please check to make sure the molecule you want exists on HITRAN. ----- ")
        else:
            break
    
    while True:
        try:
            isotopologue_ID = int(input("What is the isotopologue ID of the isotopologue you would like to download? Enter '1' for the most abundant isotopologue, '2' for next most abundant, etc. More info found here: https://hitran.org/lbl/# \n"))
            partitionSum(molecule_id_number, isotopologue_ID, [70, 80], step = 1.0)
        except ValueError:
            print("\n ----- Please enter an integer for the isotopologue ID number -----")
        except KeyError:
            print("\n ----- This molecule/isotopologue ID combination is not valid. Please check to make sure the combo you want exists on HITRAN. ----- ")
        else:
            return molecule_id_number, isotopologue_ID


def create_pf(mol_ID, iso_ID, folder, T_min = 70, T_max = 3001, step = 1.0):
    """
    Create partition function file using the partitionSum() function already in hapi

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
    J_lower_all, gamma_air, n_air = (np.array([]) for _ in range(3))
    gamma_air_avg, n_air_avg = (np.array([]) for _ in range(2))
    
    for file in os.listdir(input_dir):
        if file.endswith('.h5'):
            with h5py.File(input_dir + file, 'r') as hdf:
                
                # Populate the arrays by reading in each hdf5 file
                J_lower_all = np.append(J_lower_all, np.array(hdf.get('Lower State J')))
                gamma_air = np.append(gamma_air, np.array(hdf.get('Air Broadened Width')))
                n_air = np.append(n_air, np.array(hdf.get('Temperature Dependence of Air Broadening')))
            
    J_sorted = np.sort(np.unique(J_lower_all))
        
    for i in range(len(J_sorted)):
        
        gamma_air_i = np.mean(gamma_air[np.where(J_lower_all == J_sorted[i])])
        n_air_i = np.mean(n_air[np.where(J_lower_all == J_sorted[i])])
        gamma_air_avg = np.append(gamma_air_avg, gamma_air_i)
        n_air_avg = np.append(n_air_avg, n_air_i)
        
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
                       
    output_folder = download.create_directories(mol_ID = molecule, iso_ID = isotopologue,
                                                database = 'HITRAN')
    create_pf(molecule, isotopologue, output_folder)
    download.download_HITRAN_line_list(molecule, isotopologue, output_folder)
    for file in os.listdir(output_folder):
        if file.endswith('.data'):
            download.convert_to_hdf(file = (output_folder + file), mol_ID = molecule, 
                                    iso_ID = isotopologue, database = 'HITRAN')
    create_air_broad(output_folder)
