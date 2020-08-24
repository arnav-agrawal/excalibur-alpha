#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:16:59 2020

@author: arnav
"""

import os
import numpy as np
import pandas as pd
import re
import time
import copy
import requests
import sys
import h5py
from bs4 import BeautifulSoup
from scipy.interpolate import UnivariateSpline as Interp
from hapi import molecularMass, moleculeName, isotopologueName


from excalibur.Voigt import Voigt_width, Generate_Voigt_grid_molecules, gamma_L_VALD, gamma_L_impact, analytic_alkali
from excalibur.calculations import find_index, prior_index, bin_cross_section_atom, bin_cross_section_molecule
from excalibur.calculations import produce_total_cross_section_EXOMOL, produce_total_cross_section_HITRAN
from excalibur.calculations import produce_total_cross_section_VALD_atom, produce_total_cross_section_VALD_molecule

from excalibur.constants import nu_refer, gamma_0, n_L, P_ref, T_ref
from excalibur.constants import c, kb, h, m_e, c2, u, pi

from excalibur.Download_ExoMol import get_default_iso, get_default_linelist


def create_id_dict():
    """
    Create a dictionary that maps molecules to their HITRAN molecule ID number found here: https://hitran.org/lbl/

    Returns
    -------
    molecule_dict : dict
        Dictionary mapping a given molecule to its HITRAN ID number.

    """
    
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

def parse_directory(directory):
    """
    Determine which molecule and linelist this directory contains data for (assumes data was downloaded using our script)

    Parameters
    ----------
    directory : String
        Local directory containing the line list file[s], broadening data, and partition function

    Returns
    -------
    molecule : String
        Molecule which the cross-section is being calculated for.
    linelist : String
        Line list which the cross-section is being calculated for.

    """
    
    directory_name = os.path.abspath(directory)
    database = os.path.basename(directory_name)
    directory_name = os.path.dirname(directory_name)
    molecule = os.path.basename(directory_name)
    same_molecule = copy.deepcopy(molecule)  # Make a copy of the string because we'll need it for the isotopologue
    molecule = re.sub('[  ~].+', '', molecule)  # Keep molecule part of the folder name
    isotopologue = re.sub('.+[  ~]', '', same_molecule) # Keep isotope part of the folder name
    
    return molecule, isotopologue, database
    
    
def check_molecule(molecule):
    """
    Check if the given string is a molecule

    Parameters
    ----------
    molecule : String
        Molecular formula.

    Returns
    -------
    True if the given string is a molecule, false otherwise (if it is an atom).

    """
    match = re.match('^[A-Z]{1}[a-z]?$', molecule)     # Matches a string containing only 1 capital letter followed by 0 or 1 lower case letters
    
    if match: return False   # If our 'molecule' matches the pattern, it is really an atom
    else: return True        # We did not get a match, therefore must have a molecule


def mass(molecule, isotopologue, linelist):
    """
    Determine the mass of a given molecule-isotopologue combination

    Parameters
    ----------
    molecule : String
        Molecule we are calculating the mass for.
    isotopologue : String
        Isotopologue of this molecule we are calculating the mass for.
    linelist : String
        The line list this molecule's cross-section will later be calculated for. Used to 
        identify between ExoMol, HITRAN/HITEMP, and VALD

    Returns
    -------
    int
        Mass of the given molecule-isotopologue combination.

    """
    
    if linelist == 'hitran' or linelist == 'hitemp':
        mol_ID = 1
        while moleculeName(mol_ID) != molecule:
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
        
        return mass_dict.get(molecule)

        
    else:
        isotopologue = isotopologue.replace('(', '')
        isotopologue = isotopologue.replace(')', '')
        url = 'http://exomol.com/data/molecules/' + molecule + '/' + isotopologue + '/' + linelist + '/'
        
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
        
    
def load_ExoMol(input_directory):
    """
    Read in the '.states' file downloaded from ExoMol

    Parameters
    ----------
    input_directory : String
        Directory that contains all the ExoMol downloaded files for the desired molecule/
        isotopologue/line list.

    Returns
    -------
    E : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    J : TYPE
        DESCRIPTION.

    """
    
    # Read in states file (EXOMOL only)
    states_file_name = [filename for filename in os.listdir(input_directory) if filename.endswith('.states')]
    states_file = pd.read_csv(input_directory + states_file_name[0], sep = '\s+', header=None, usecols=[0,1,2,3])
    E = np.array(states_file[1])
    g = np.array(states_file[2])
    J = np.array(states_file[3]).astype(np.int64)
    
    del states_file  # Delete file to free up memory    
    
    return E, g, J

def load_VALD(input_directory, molecule):
    
    fname = [file for file in os.listdir(input_directory) if file.endswith('.h5')][0]  # The directory should only have one .h5 file containing the line list
    
    with h5py.File(input_directory + fname, 'r') as hdf:
        nu_0 = np.array(hdf.get('nu'))
        gf = np.power(10.0, np.array(hdf.get('Log gf')))
        E_low = np.array(hdf.get('E lower'))
        E_up = np.array(hdf.get('E upper'))
        J_low = np.array(hdf.get('J lower'))
        gamma_nat = np.power(10.0, np.array(hdf.get('Log gamma nat')))
        gamma_vdw = np.power(10.0, np.array(hdf.get('Log gamma vdw')))
        
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
    gamma_nat = gamma_nat[order]
    gamma_vdw = gamma_vdw[order]
    
    if alkali:
        l_low = l_low[order]
        l_up = l_up[order]
    
    return nu_0, gf, E_low, E_up, J_low, l_low, l_up, gamma_nat, gamma_vdw, alkali
        
    return


def load_pf(input_directory):

    """
    Read in the downloaded partition functions 

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

    """
    print("Loading partition functions")
    pf_file_name = [filename for filename in os.listdir(input_directory) if filename.endswith('.pf')]
    pf_file = pd.read_csv(input_directory + pf_file_name[0], sep= ' ', header=None, skiprows=1)
    T_pf_raw = np.array(pf_file[0]).astype(np.float64)
    Q_raw = np.array(pf_file[1])

    del pf_file   # Delete file to free up memory
    
    return T_pf_raw, Q_raw


def det_broad(input_directory):
    """
    Determine the type of broadening that should be used in the case that the user specifies
    'default' broadening. Order of preference is: 1) H2-He, 2) air, 3) Burrows

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    broadening : String
        The type of broadening being used.

    """
    if 'H2.broad' in os.listdir(input_directory) and 'He.broad' in os.listdir(input_directory):
        broadening = 'H2-He'
        
    elif 'air.broad' in os.listdir(input_directory):
        broadening = 'air'
        
    else:
        broadening = 'Burrows'
        if not 'Burrows.broad' in os.listdir(input_directory):
            create_Burrows(input_directory)
        
    return broadening


def create_Burrows(input_directory):
    """
    Create Burrows broadening file (as specified in Eq. 15 of Sharp and Burrows (2007), and add 
    it to the input_directory

    Parameters
    ----------
    input_directory : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    burrows_file = input_directory + 'Burrows.broad'
    J = np.arange(31.0)
    gamma_L_0 = np.zeros(31)
    N_L = np.zeros(31)
    
    for i in range(len(J)):
        gamma_L_0[i] = (0.1 - min(J[i], 30) * 0.002) / (1.01325 * 2) # Convert from cm^-1 / atm -> cm^-1 / bar and take width at half-max

    f_out = open(burrows_file, 'w')
    f_out.write('J | gamma_L_0 | n_L \n')
    for i in range(len(J)):
        f_out.write('%.1f %.4f %.3f \n' %(J[i], gamma_L_0[i], N_L[i]))
        
    f_out.close()


def read_H2_He(input_directory):
    
    # Read in H2 broadening file
    broad_file_H2 = pd.read_csv(input_directory + 'H2.broad', sep = ' ', header=None, skiprows=1)
    J_max_H2 = int(np.max(np.array(broad_file_H2[0])))
    gamma_0_H2 = np.array(broad_file_H2[1])
    n_L_H2 = np.array(broad_file_H2[2])
    
    # Read in He broadening file
    broad_file_He = pd.read_csv(input_directory + 'He.broad', sep = ' ', header=None, skiprows=1)
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
                
    del broad_file_H2, broad_file_He   # Delete files to free up memory
    
    return J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He
    
    
def read_air(input_directory):
    
    # Read in air broadening file
    broad_file_air = pd.read_csv(input_directory + 'air.broad', sep = ' ', header=None, skiprows = 1)
    J_max = int(np.max(np.array(broad_file_air[0])))
    gamma_0_air = np.array(broad_file_air[1])
    n_L_air = np.array(broad_file_air[2])
    
    del broad_file_air   # Delete file to free up memory  
    
    return J_max, gamma_0_air, n_L_air

def read_custom(input_directory):
    # Read in custom broadening file
    broad_file_custom = pd.read_csv(input_directory + 'custom.broad', sep = ' ', header=None, skiprows = 1)
    J_max = int(np.max(np.array(broad_file_custom[0])))
    gamma_0_air = np.array(broad_file_custom[1])
    n_L_air = np.array(broad_file_custom[2])
    
    del broad_file_custom   # Delete file to free up memory  
    
    return J_max, gamma_0_air, n_L_air
    

def read_Burrows(input_directory):
    
    # Read in Burrows broadening file
    broad_file_Burrows = pd.read_csv(input_directory + 'Burrows.broad', sep = ' ', header=None, skiprows=1)
    J_max = int(np.max(np.array(broad_file_Burrows[0])))
    gamma_0_Burrows = np.array(broad_file_Burrows[1])
    #n_L_Burrows = np.array(broad_file_Burrows[2])       # Not really needed, as temperature exponent = 0 for all J''
    
    del broad_file_Burrows   # Delete file to free up memory
    
    return J_max, gamma_0_Burrows


def interpolate_pf(T_pf_raw, Q_raw, T, T_ref):
    
    #***** Interpolate (and extrapolate) partition function to a fine grid *****#
    pf_spline = Interp(T_pf_raw, Q_raw, k=5)
    T_pf_fine = np.linspace(1.0, 10000.0, 9999)       # New temperature grid (extrapolated to 4000K)
    Q_fine = pf_spline(T_pf_fine)                    # Extrapolated partition function
    i_T = np.argmin(np.abs(T_pf_fine - T))           # Index of partition function temperature array closest to desired temperature
    i_T_ref = np.argmin(np.abs(T_pf_fine - T_ref))   # Index of partition function temperature array closest to reference temperature
    Q_T = Q_fine[i_T]                                # Partition function at given temperature, Q(T)
    Q_T_ref = Q_fine[i_T_ref]                        # Partition function at reference temperature, Q(T_ref)
    
    return Q_T, Q_T_ref

def read_pressure_broadening_atom(species, nu_0, gf, E_low, E_up, J_low, l_low, l_up, gamma_nat, gamma_vdw, alkali, m):
    
    if alkali:  # Special treatments for alkali van der waals widths
                
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
                        
                # Remove transitions where Hydrogenic approximation breaks down
          #      accept_condition = np.where(gamma_0_H2 != -1.0)  # Transitions without a VALD broadening value
                        
          #      nu_0 = nu_0[accept_condition]
          #      gf = gf[accept_condition]
          #      E_low = E_low[accept_condition]
          #      E_up = E_up[accept_condition]
          #      J_low = J_low[accept_condition]
          #      l_low = l_low[accept_condition]
          #      l_up = l_up[accept_condition]
          #      n_L_H2 = n_L_H2[accept_condition]
          #      n_L_He = n_L_He[accept_condition]
          #      gamma_0_He = gamma_0_He[accept_condition]
          #      gamma_0_H2 = gamma_0_H2[accept_condition]
                
    else:  # For non-alkali species
                
          #      accept_condition = np.where(log_gamma_vdw != 0.0)  # Transitions without a VALD broadening value 
                
          #      log_gamma_vdw = log_gamma_vdw[accept_condition]
          #      nu_0 = nu_0[accept_condition]
          #      gf = gf[accept_condition]
          #      E_low = E_low[accept_condition]
          #      E_up = E_up[accept_condition]
          #      J_low = J_low[accept_condition]
          #      l_low = l_low[accept_condition]
          #      l_up = l_up[accept_condition]
                
        gamma_0_H2, n_L_H2 = gamma_L_VALD(gamma_vdw, (m/u), 'H2')
        gamma_0_He, n_L_He = gamma_L_VALD(gamma_vdw, (m/u), 'He')
        
    return gamma_0_H2, n_L_H2, gamma_0_He, n_L_He


def compute_H2_He_broadening(gamma_0_H2, T_ref, T, n_L_H2, P, P_ref, X_H2, gamma_0_He, n_L_He, X_He):
    gamma = (gamma_0_H2 * np.power((T_ref/T), n_L_H2) * (P/P_ref) * X_H2 +   # H2+He Lorentzian HWHM for given T, P, and J (ang. mom.)
             gamma_0_He * np.power((T_ref/T), n_L_He) * (P/P_ref) * X_He)    # Note that these are only a function of J''

    return gamma
    
def compute_air_broadening(gamma_0_air, T_ref, T, n_L_air, P, P_ref):
    gamma = (gamma_0_air * np.power((T_ref/T), n_L_air) * (P/P_ref))      # Air-broadened Lorentzian HWHM for given T, P, and J (ang. mom.)

    return gamma
    
def compute_Burrows_broadening(gamma_0_Burrows, P, P_ref):
    gamma = (gamma_0_Burrows * (P/P_ref))      # Equation (15) in Sharp & Burrows (2007)  
    
    return gamma

def create_wavelength_grid_atom(T, m, gamma, nu_0, Voigt_sub_spacing, dnu_out, nu_out_min, nu_out_max, 
                                Voigt_cutoff, cut_max, molecule):
    
    #Define nu_out and nu_min based on the output grid
    nu_min = 1
    nu_max = nu_out_max + 1000
    
    
    # First, we need to find values of gamma_V for reference wavenumber (1000 cm^-1)
    alpha_ref = np.sqrt(2.0*kb*T*np.log(2)/m) * (np.array(nu_refer[2])/c) # Doppler HWHM at reference wavenumber
    gamma_ref = np.min(gamma)                                           # Find minimum value of Lorentzian HWHM
    gamma_V_ref = Voigt_width(gamma_ref, alpha_ref)                     # Reference Voigt width
    
    # Calculate Voigt width for each transition
    alpha = np.sqrt(2.0*kb*T*np.log(2)/m) * (np.array(nu_0)/c)   # Doppler HWHM for each transition
    gamma_V = Voigt_width(gamma, alpha)
    
    # Now compute properties of computational (fine) and output (coarse) wavenumber grid
    
    # Wavenumber spacing of of computational grid (smallest of gamma_V_ref/6 or 0.01cm^-1)
    dnu_fine = np.minimum(gamma_V_ref*Voigt_sub_spacing, dnu_out)      
    
    # Number of points on fine grid (rounded)
    N_points_fine = int((nu_max-nu_min)/dnu_fine + 1)
    
    # Adjust dnu_fine slightly to match an exact integer number of grid spaces
    dnu_fine = (nu_max-nu_min)/(N_points_fine - 1)
    
    cutoffs = np.zeros(len(nu_0))   # Line wing cutoffs for each line
    
    # Line cutoffs at @ min(500 gamma_V, 1000cm^-1)
    for i in range(len(nu_0)):
        cutoffs[i] = dnu_fine * (int((Voigt_cutoff*gamma_V[i])/dnu_fine))

        if (cutoffs[i] >= cut_max): cutoffs[i] = cut_max
                
        # Special cases for alkali resonant lines
        if   ((molecule == 'Na') and (int(nu_0[i]) in [16978, 16960])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        elif ((molecule == 'K') and  (int(nu_0[i]) in [13046, 12988])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        #elif ((species == 'Li') and (int(nu_0[i]) in [14908, 14907])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        #elif ((species == 'Rb') and (int(nu_0[i]) in [12820, 12582])): cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
        #elif ((species == 'Cs') and (int(nu_0[i]) in [11735, 9975])):  cutoffs[i] = 9000.0   # Cutoff @ +/- 9000 cm^-1
            
    # Calculate detuning frequencies for Na and K resonance lines
    if (molecule == 'Na'): nu_detune = 30.0 * np.power((T/500.0), 0.6)
    elif (molecule == 'K'): nu_detune = 20.0 * np.power((T/500.0), 0.6)
    else: nu_detune = cut_max
    
    # Evaluate number of frequency points for each Voigt function up to cutoff (one tail)
    N_Voigt_points = ((cutoffs/dnu_fine).astype(np.int64)) + 1  
    
    # Define start and end points of fine grid
    #nu_fine = np.linspace(nu_min, nu_max, N_points_fine)
    nu_fine_start = nu_min
    nu_fine_end = nu_max
    
    # Initialise output grid
    N_points_out = int((nu_out_max-nu_out_min)/dnu_out + 1)     # Number of points on coarse grid (uniform)
    nu_out = np.linspace(nu_out_min, nu_out_max, N_points_out)  # Create coarse (output) grid
    
    # Initialise cross section arrays on each grid
    sigma_fine = np.zeros(N_points_fine)    # Computational (fine) grid
    sigma_out = np.zeros(N_points_out)      # Coarse (output) grid
    
    return (sigma_fine, nu_detune, N_points_fine, N_Voigt_points, alpha, cutoffs, nu_min, nu_max, 
            nu_fine_start, nu_fine_end, nu_out, sigma_out, N_points_out)

def create_wavelength_grid_molecule(nu_ref, m, T, gamma, Voigt_sub_spacing, dnu_out, cut_max, Voigt_cutoff, nu_out_max, nu_out_min, N_alpha_samples):
    #nu_min = max(1.0, (nu_out_min - cut_max))
    #nu_max = nu_out_max + cut_max
    
    nu_min = 1
    nu_max = nu_out_max + 1000

    # First, we need to find values of gamma_V for reference wavenumbers (100, 1000, 10000 cm^-1)
    nu_ref = np.array(nu_ref)   # The reference wavenumbers need to be a numpy array
    alpha_ref = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_ref/c)    # Doppler HWHM at reference wavenumbers 
    gamma_ref = np.min(gamma)                                           # Find minimum value of Lorentzian HWHM
    gamma_V_ref = Voigt_width(gamma_ref, alpha_ref)                     # Reference Voigt widths
    
    # Now compute properties of computational (fine) and output (coarse) wavenumber grids
            
    # Wavenumber spacing of three regions of computational grid (smallest of gamma_V_ref/6 or 0.01cm^-1)
    dnu_fine = np.minimum(gamma_V_ref*Voigt_sub_spacing, dnu_out*np.ones(3))
    
    # Number of points on fine grid (three regions, rounded)
    N_points_fine_1 = int((nu_ref[1]-nu_min)/dnu_fine[0])
    N_points_fine_2 = int((nu_ref[2]-nu_ref[1])/dnu_fine[1])
    N_points_fine_3 = int((nu_max-nu_ref[2])/dnu_fine[2] + 1)
    N_points_fine = N_points_fine_1 + N_points_fine_2 + N_points_fine_3
    
    # Adjust dnu_fine slightly to match an exact integer number of grid spaces
    dnu_fine[0] = (nu_ref[1]-nu_min)/N_points_fine_1
    dnu_fine[1] = (nu_ref[2]-nu_ref[1])/N_points_fine_2
    dnu_fine[2] = (nu_max-nu_ref[2])/(N_points_fine_3 - 1)
    cutoffs = np.zeros(len(dnu_fine))   # Line wing cutoffs in three regions
    
    # Line cutoffs at @ min(500 gamma_V, 30cm^-1)
    for i in range(len(dnu_fine)):     
        # If grid spacing below maximum dnu (0.01), set to 3000 dnu (~ 500 gamma_V)
        if (dnu_fine[i] < dnu_out):
            cutoffs[i] = Voigt_cutoff*(1.0/Voigt_sub_spacing)*dnu_fine[i] 
            
        # If at maximum dnu (0.01), set to min(500 gamma_V, 30cm^-1)
        else:
            cutoffs[i] = dnu_fine[i] * int(Voigt_cutoff*gamma_V_ref[i]/dnu_fine[i])
            if (cutoffs[i] >= cut_max): cutoffs[i] = cut_max
            
    # Define start and end points of each fine grid
    nu_fine_1_start = nu_min
    nu_fine_1_end = (nu_ref[1] - dnu_fine[0])
    nu_fine_2_start = nu_ref[1]
    nu_fine_2_end = (nu_ref[2] - dnu_fine[1])
    nu_fine_3_start = nu_ref[2]
    nu_fine_3_end = nu_max
    
    #nu_fine_1 = np.linspace(nu_min, nu_ref[1], N_points_fine_1, endpoint=False)
    #nu_fine_2 = np.linspace(nu_ref[1], nu_ref[2], N_points_fine_2, endpoint=False)
    #nu_fine_3 = np.linspace(nu_ref[2], nu_max, N_points_fine_3)
    #nu_fine = np.concatenate([nu_fine_1, nu_fine_2, nu_fine_3])
    
    # Initialise output grid
    N_points_out = int((nu_out_max-nu_out_min)/dnu_out + 1)     # Number of points on coarse grid (uniform)
    nu_out = np.linspace(nu_out_min, nu_out_max, N_points_out)  # Create coarse (output) grid
        
    # Initialise cross section arrays on each grid
    sigma_fine = np.zeros(N_points_fine)    # Computational (fine) grid
    
    #sigma_out = np.zeros(shape=(N_P, N_T, N_points_out))
    
    sigma_out = np.zeros(N_points_out)      # Coarse (output) grid
    
    return (sigma_fine, nu_ref, N_points_fine_1, N_points_fine_2, N_points_fine_3, dnu_fine, cutoffs,
            nu_out, sigma_out, nu_fine_1_start, nu_fine_1_end, nu_fine_2_start, nu_fine_2_end, 
            nu_fine_3_start, nu_fine_3_end, N_points_out, nu_min, nu_max, alpha_ref)
    
    
def precompute_Voigt_profiles(nu_ref, nu_max, N_alpha_samples, T, m, cutoffs, dnu_fine, gamma,
                              alpha_ref):
    
    #***** Pre-compute Voigt function array for molecules *****#
    
    print('Pre-computing Voigt profiles...')
    
    t1 = time.perf_counter()
        
    # First, create an array of values of alpha to approximate true values of alpha (N=500 log-spaced => max error of 0.5%)
    nu_sampled = np.logspace(np.log10(nu_ref[0]), np.log10(nu_max), N_alpha_samples)
    alpha_sampled = np.sqrt(2.0*kb*T*np.log(2)/m) * (nu_sampled/c)
    
    # Evaluate number of frequency points for each Voigt function in each spectral region - up to cutoff @ min(500 gamma_V, 30cm^-1)
    N_Voigt_points = ((cutoffs/dnu_fine).astype(np.int64)) + 1  
            
    # Pre-compute and store Voigt functions and first derivatives wrt alpha 
    Voigt_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))    # For H2O: V(51,500,3001)
    dV_da_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))    # For H2O: V(51,500,3001)
    dV_dnu_arr = np.zeros(shape=(len(gamma), len(alpha_sampled), np.max(N_Voigt_points)))   # For H2O: V(51,500,3001)
    Generate_Voigt_grid_molecules(Voigt_arr, dV_da_arr, dV_dnu_arr, gamma, alpha_sampled, alpha_ref, cutoffs, N_Voigt_points)
        
    t2 = time.perf_counter()
    total1 = t2-t1
            
    print('Voigt profiles computed in ' + str(total1) + ' s')       

    return nu_sampled, alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr, N_Voigt_points
        
        
def write_output_file(cluster_run, output_directory, molecule, T, log_P, nu_out, sigma_out):
    #***** Now write output files *****#
            
    f = open(output_directory + str(molecule) + '_T' + str(T) + 'K_log_P' + str(log_P) + '_sigma.txt','w')
                    
    for i in range(len(nu_out)):
        f.write('%.8f %.8e \n' %(nu_out[i], sigma_out[i]))
                        
    f.close()
    
    return nu_out, sigma_out
    
def replace_iso_name(iso_name):
    """
    Replace the isotopologue name generated by HITRAN/HITEMP to match the isotopologue name convention
    used by ExoMol.

    Parameters
    ----------
    iso_name : String
        Name of the isotopologue that is to be replaced.

    Returns
    -------
    iso_name : String
        A differently formatted version of the isotopologue name passed in. Matches ExoMol format.

    """
    # 'H' not followed by lower case letter needs to become '(1H)'
    iso_name = re.sub('H(?![a-z])', '(1H)', iso_name)
    
    # Number of that atom needs to be enclosed by parentheses ... so '(1H)2' becomes '(1H2)'
    matches = re.findall('[)][0-9]{1}', iso_name)
    for match in matches:
        number = re.findall('[0-9]{1}', match)
        iso_name = re.sub('[)][0-9]{1}', number[0] + ')', iso_name)
    
    # replace all ')(' with '-'
    iso_name = iso_name.replace(')(', '-')   
    
    return iso_name
    
    
def find_input_dir(input_dir, database, molecule, isotope, ionization_state, linelist):
    """
    Find the directory on a user's machine that contains the data needed to create a cross-section

    Parameters
    ----------
    input_dir : String
        'Prefix' of the directory containing the line list files. If the files were downloaded
        using our Download_Line_List.py script, input_dir will end in '/input'
    database : String
        Database the line list was downloaded from.
    molecule : String
        Molecule for which the cross-section is created.
    isotope : String
        Isotopologue of the molecule for which the cross-section was created.
    linelist : String
        Line list that is being used. HITRAN/HITEMP/VALD used as the line list name for these 
        databases respectively. ExoMol has its own named line lists.

    Returns
    -------
    input_directory : TYPE
        DESCRIPTION.

    """
    
    if isotope == 'default':
        if database == 'exomol':
            isotope = '(' + get_default_iso(molecule) + ')'
        if database == 'hitran' or database == 'hitemp':
            molecule_dict = create_id_dict()
            mol_id = molecule_dict.get(molecule)
            isotope = isotopologueName(mol_id, 1)
            isotope = replace_iso_name(isotope)
    
    if database == 'vald':
        isotope = ''
        for i in range(ionization_state):  # Make it easier to code later in the function by just assigning ionization state to isotope even though they're not the same thing
            isotope += 'I'
        isotope = '(' + isotope + ')'
    
    if linelist == 'default':
        if database == 'exomol':
            temp_isotope = re.sub('[(]|[)]', '', isotope)
            linelist = get_default_linelist(molecule, temp_isotope)
        if database == 'hitran':
            linelist = 'HITRAN'
        if database == 'hitemp':
            linelist = 'HITEMP'
        if database == 'vald':
            linelist = 'VALD'
            
    input_directory = input_dir + '/' + molecule + '  ~  ' + isotope + '/' + linelist + '/'
    
    if os.path.exists(input_directory):
        return input_directory
    
    else:
        print("You don't seem to have a local folder with the parameters you entered.\n") 
        
        if not os.path.exists(input_dir + '/'):
            print("----- You entered an invalid input directory into the cross_section() function. Please try again. -----")
            sys.exit(0)
        
        elif not os.path.exists(input_dir + '/' + molecule + '  ~  ' + isotope + '/'):
            print("----- There was an error with the molecule + isotope you entered. Here are the available options: -----\n")
            for folder in os.listdir(input_dir + '/'):
                if not folder.startswith('.'):
                    print(folder)
            sys.exit(0)
        
        else:
            print("There was an error with the line list. These are the linelists available: \n")
            for folder in os.listdir(input_dir + '/' + molecule + '  ~  ' + isotope + '/'):
                if not folder.startswith('.'):
                    print(folder)
            sys.exit(0)

    
def create_cross_section(input_dir, database, molecule, log_pressure, temperature, isotope = 'default', 
                         ionization_state = 1, linelist = 'default', cluster_run = False, nu_out_min = 200, 
                         nu_out_max = 25000, dnu_out = 0.01, pressure_broadening = 'default', X_H2 = 0.85, 
                         X_He = 0.15, Voigt_cutoff = 500, Voigt_sub_spacing = (1.0/6.0), N_alpha_samples = 500, 
                         S_cut = 1.0e-100, cut_max = 30.0, **kwargs):
    
    print("Beginning cross-section computations...")
    
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
    input_directory = find_input_dir(input_dir, database, molecule, isotope, ionization_state, linelist)
    
    
    # Use the input directory to define these right at the start
    molecule, isotopologue, database = parse_directory(input_directory)
    if database.lower() != 'hitran' and database.lower() != 'hitemp' and database.lower() != 'vald':
        linelist = database
        database = 'exomol'
    else:
        database = database.lower()
        linelist = database
    
    linelist_files = [filename for filename in os.listdir(input_directory) if filename.endswith('.h5')]
    
    if database == 'exomol':
        print("Loading ExoMol format")
        E, g, J = load_ExoMol(input_directory)
    
    if database == 'hitran':
        print("Loading HITRAN format")
        # Nothing else required
    
    if database == 'hitemp':
        print("Loading HITEMP format")
        # Nothing else required
        
    if database == 'vald':
        print("Loading VALD format")
        nu_0, gf, E_low, E_up, J_low, l_low, l_up, gamma_nat, gamma_vdw, alkali = load_VALD(input_directory, molecule)
        
    T_pf_raw, Q_raw = load_pf(input_directory)
    
    # Find mass of the molecule
    m = mass(molecule, isotopologue, linelist) * u


    is_molecule = check_molecule(molecule)
    
    if is_molecule and pressure_broadening == 'default':
        broadening = det_broad(input_directory)
        if broadening == 'H2-He':
            J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He = read_H2_He(input_directory)
            
        elif broadening == 'air':
            J_max, gamma_0_air, n_L_air = read_air(input_directory)
            
        elif broadening == 'Burrows':
            J_max, gamma_0_Burrows = read_Burrows(input_directory)
            
    elif is_molecule and pressure_broadening != 'default':
        broadening = pressure_broadening
        if broadening == 'H2-He' and 'H2.broad' in os.listdir(input_directory) and 'He.broad' in os.listdir(input_directory):
            J_max, gamma_0_H2, n_L_H2, gamma_0_He, n_L_He = read_H2_He(input_directory)
        
        elif broadening == 'air' and 'air.broad' in os.listdir(input_directory):
            broadening = 'air'
            J_max, gamma_0_air, n_L_air = read_air(input_directory)
            
        elif broadening == 'Burrows':
            create_Burrows(input_directory)
            J_max, gamma_0_Burrows = read_Burrows(input_directory)
            
        elif broadening == 'custom' and 'custom.broad' in os.listdir(input_directory):
            J_max, gamma_0_air, n_L_air = read_custom(input_directory)
        
        elif broadening == 'fixed':
            J_max = 0
        
        else:
            print("\nYou did not enter a valid type of pressure broadening. Please try again.")
            sys.exit(0)
            
    else:
        if pressure_broadening != 'default' and pressure_broadening != 'H2-He':
            print("\nYou did not specify a valid choice of pressure broadening. For atoms, the only option is 'H2-He', so we will continue by using that." )
        broadening = 'H2-He'
        gamma_0_H2, gamma_0_He, n_L_H2, n_L_He = read_pressure_broadening_atom(molecule, nu_0, gf, E_low, E_up, 
                                                                               J_low, l_low, l_up, gamma_nat, 
                                                                               gamma_vdw, alkali, m)
        

    # Start clock for timing program
    t_start = time.perf_counter()
    
    #***** Load pressure and temperature for this calculation *****#
    P_arr = np.power(10.0, log_pressure) 
    log_P_arr = np.array(log_pressure)        # log_10 (Pressure/bar)
    T_arr = np.array(temperature)         # Temperature (K)
    
    # If running on Condor
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
        
        #log_P = log_P_arr[idx_PT//len(T_arr)]   # Log10 atmospheric pressure (bar)  # Pressure combinations mapped before temperature
        P = P_arr[idx_PT//len(T_arr)]   # Atmospheric pressure (bar)
        T = T_arr[idx_PT%len(T_arr)]   # Atmospheric temperature (K)
        
        N_P = 1
        N_T = 1
        
    else:
        
        N_P = len(log_P_arr)
        N_T = len(T_arr)
    

    for p in range(N_P):
        for t in range(N_T):
            if (cluster_run == False):
                
                P = P_arr[p]   # Atmospheric pressure (bar)
                T = T_arr[t]   # Atmospheric temperature (K)
            
            Q_T, Q_T_ref = interpolate_pf(T_pf_raw, Q_raw, T, T_ref)
            
            if is_molecule: # Molecules
                
                if broadening == 'H2-He':
                    gamma = compute_H2_He_broadening(gamma_0_H2, T_ref, T, n_L_H2, P, P_ref, X_H2, gamma_0_He, n_L_He, X_He)
                
                elif broadening == 'air':
                    gamma = compute_air_broadening(gamma_0_air, T_ref, T, n_L_air, P, P_ref)
                    
                elif broadening == 'Burrows':
                    gamma = compute_Burrows_broadening(gamma_0_Burrows, P, P_ref)
                    
                elif broadening == 'custom':  # Computation step is the same as for air broadening
                    gamma = compute_air_broadening(gamma_0_air, T_ref, T, n_L_air, P, P_ref)
                    
                elif broadening == 'fixed':
                    gamma = np.array([(gamma_0 * np.power((T_ref/T), n_L) * (P/P_ref))])  # Fixed Lorentizian HWHM (1 element array)
                    
                (sigma_fine, nu_ref, N_points_fine_1, N_points_fine_2, 
                 N_points_fine_3, dnu_fine, cutoffs, nu_out, sigma_out, 
                 nu_fine_1_start, nu_fine_1_end, nu_fine_2_start, 
                 nu_fine_2_end, nu_fine_3_start, nu_fine_3_end, 
                 N_points_out, nu_min, nu_max, alpha_ref) = create_wavelength_grid_molecule(nu_refer, m, T, gamma, Voigt_sub_spacing, 
                                                                                            dnu_out, cut_max, Voigt_cutoff, nu_out_max, 
                                                                                            nu_out_min, N_alpha_samples)
                                                                                               
                (nu_sampled, alpha_sampled, Voigt_arr, 
                 dV_da_arr, dV_dnu_arr, N_Voigt_points) = precompute_Voigt_profiles(nu_ref, nu_max, N_alpha_samples, T, m, cutoffs, dnu_fine, gamma, alpha_ref)                                                                            
                
            else:  # Atoms
                
                gamma = compute_H2_He_broadening(gamma_0_H2, T_ref, T, n_L_H2, P, P_ref, X_H2, gamma_0_He, n_L_He, X_He)
                gamma += ((1.0/(4.0*np.pi*(100.0*c))) * gamma_nat)  # Add natural line widths
            
                
                (sigma_fine, nu_detune, N_points_fine, N_Voigt_points, 
                 alpha, cutoffs, nu_min, nu_max, nu_fine_start, 
                 nu_fine_end, nu_out, sigma_out, N_points_out) = create_wavelength_grid_atom(T, m, gamma, nu_0, Voigt_sub_spacing, 
                                                                                             dnu_out, nu_out_min, nu_out_max, 
                                                                                             Voigt_cutoff, cut_max, molecule)
                                                                                                                                        
                
            print("Pre-computation complete")
                
            
            print('Generating cross section for ' + molecule + ' at P = ' + str(P) + ' bar, T = ' + str(T) + ' K')
            
            if database == 'exomol':                
                produce_total_cross_section_EXOMOL(linelist_files, input_directory, sigma_fine,
                                                   nu_sampled, nu_ref, m, T, Q_T, N_points_fine_1,
                                                   N_points_fine_2, N_points_fine_3, dnu_fine,
                                                   N_Voigt_points, cutoffs, g, E, J, J_max, 
                                                   alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr,
                                                   nu_min, nu_max, S_cut)
                
            elif database == 'hitran':
                produce_total_cross_section_HITRAN(linelist_files, input_directory, sigma_fine,
                                                   nu_sampled, nu_ref, m, T, Q_T, Q_T_ref,
                                                   N_points_fine_1, N_points_fine_2, N_points_fine_3,
                                                   dnu_fine, N_Voigt_points, cutoffs, J_max, 
                                                   alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr,
                                                   nu_min, nu_max, S_cut)
                
            elif database == 'hitemp':
                produce_total_cross_section_HITRAN(linelist_files, input_directory, sigma_fine,
                                                   nu_sampled, nu_ref, m, T, Q_T, Q_T_ref,
                                                   N_points_fine_1, N_points_fine_2, N_points_fine_3,
                                                   dnu_fine, N_Voigt_points, cutoffs, J_max, 
                                                   alpha_sampled, Voigt_arr, dV_da_arr, dV_dnu_arr,
                                                   nu_min, nu_max, S_cut)
                
        #        nu_0, gf, E_low, E_up, J_low, l_low, l_up, gamma_nat, gamma_vdw, alkali 
                
            elif database == 'vald':
                produce_total_cross_section_VALD_atom(sigma_fine, nu_0, nu_detune, E_low, gf, m, T, Q_T,
                                                      N_points_fine, N_Voigt_points, alpha, gamma, cutoffs,
                                                      nu_min, nu_max, S_cut, molecule)
                    
            
            # Now bin cross section to output grid    
            print('Binning cross section to output grid...')
        
            if is_molecule:
                bin_cross_section_molecule(sigma_fine, sigma_out, N_points_fine_1, N_points_fine_2,
                                           N_points_fine_3, nu_ref, nu_fine_1_start, nu_fine_1_end,
                                           nu_fine_2_start, nu_fine_2_end, nu_fine_3_start, nu_fine_3_end,
                                           nu_out, N_points_out, 0, nu_min, nu_max)
            
            
            else:
                bin_cross_section_atom(sigma_fine, sigma_out, nu_fine_start, 
                                       nu_fine_end, nu_out, N_points_fine, N_points_out, 0)
            
            
            #bin_cross_section(sigma_fine, sigma_out_log, nu_fine_1, nu_fine_2, nu_fine_3, nu_out, N_points_out, 1)
            
            # Write cross section to file
            output_directory = re.sub('/input/', '/output/', input_directory)
    
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
    
            nu, sigma = write_output_file(cluster_run, output_directory, molecule, T, log_P_arr[p], nu_out, sigma_out)
    
    t_final = time.perf_counter()
    total_final = t_final-t_start
    
    print('Total runtime: ' + str(total_final) + ' s')
    
    return nu, sigma
