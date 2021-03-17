import sys
import os
import requests
import re
import shutil
import pandas as pd
import numpy as np

import excalibur.downloader as download



def check(molecule, isotope = '', linelist = ''):
    """
    Checks if the parameters passed into the summon() function by the user are valid to use with the ExoMol database.

    Parameters
    ----------
    molecule : TYPE
        DESCRIPTION.
    isotope : TYPE
        DESCRIPTION.
    linelist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    website = "http://exomol.com/data/molecules/" + molecule + '/' + isotope + '/' + linelist + '/'
    
    try:
        response = requests.get(website)
        response.raise_for_status() # Raises HTTPError if a bad request is made (server or client error)
        
    except requests.HTTPError:
        print("\n ----- These are not valid ExoMol parameters. Please try calling the summon() function again. -----")
        sys.exit(0)
        
        
def get_default_iso(molecule):
    """
    Returns the default (most abundant on Earth) isotopologue given a molecule

    Parameters
    ----------
    molecule : String
        DESCRIPTION.

    Returns
    -------
    default_iso : String
        DESCRIPTION.

    """
    
    if molecule == 'H2_p':  # Handle H2+ on ExoMol
        default_iso = '1H-2H_p'
        return default_iso
    
    most_abundant = {'H' : 1, 'He' : 4, 'Li' : 7, 'Be' : 9, 'B' : 11, 'C' : 12, 'N' : 14, 'O' : 16, 'F' : 19, 'Ne' : 20, 
                     'Na' : 23, 'Mg' : 24, 'Al' : 27, 'Si' : 28, 'P' : 31, 'S' : 32, 'Cl' : 35, 'Ar' : 40, 'K' : 39, 'Ca' : 40,
                     'Sc' : 45, 'Ti' : 48, 'V' : 51, 'Cr' : 52, 'Mn' : 55, 'Fe' : 56, 'Co' : 59, 'Ni' : 58, 'Cu' : 63, 
                     'Zn' : 64, 'Ga' : 69, 'Ge' : 74, 'As' : 75, 'Se' : 80, 'Br' : 79, 'Kr' : 84, 'Rb' : 85, 'Sr' : 88,
                     'Y' : 89, 'Zr' : 90, 'Nb' : 93, 'Mo' : 98, 'Ru' : 102, 'Rh' : 103, 'Pd' : 106, 'Ag' : 107, 'Cd' : 114,
                     'In' : 115, 'Sn' : 120, 'Sb' : 121, 'Te' : 130, 'I' : 127, 'Xe' : 132, 'Cs' : 133, 'Ba' : 138, 'La' : 139,
                     'Ce' : 140, 'Pr' : 141, 'Nd' : 142, 'Sm' : 152, 'Eu' : 153, 'Gd' : 158, 'Tb' : 159, 'Dy' : 164, 'Ho' : 165,
                     'Er' : 166, 'Tm' : 169, 'Yb' : 174, 'Lu' : 175, 'Hf' : 180, 'Ta' : 181, 'W' : 184, 'Re' : 187, 'Os' : 192,
                     'Ir' : 193, 'Pt' : 195, 'Au' : 197, 'Hg' : 202, 'Tl' : 206, 'Pb' : 208, 'Bi' : 209, 'Th' : 232, 'Pa' : 231,
                     'Ur' : 238}
    
    default_iso = ''
    matches = re.findall('[A-Z][a-z]?[0-9]?(?:_p)?', molecule)
    num_matches = len(matches)
    for match in matches:
        num_matches -= 1
        letters = re.findall('[A-Za-z]+', match) # alphabetic characters in the molecule
        numbers = re.findall('\d', match) # numeric characters in the molecule
        ion = re.findall('(_p)', match)  # match the '_p' part if ion
        default_iso += str(most_abundant.get(letters[0])) + letters[0] 

        if numbers:
            default_iso += numbers[0]   
            
        if ion:
            default_iso += ion[0] 
            
        if num_matches != 0:
            default_iso += '-'

    return default_iso


def get_default_linelist(molecule, isotopologue):
    """
    Returns a default line list for a given ExoMol molecule and isotopologue
    These are sometimes changed as ExoMol releases better and more up-to-date line lists

    Parameters
    ----------
    molecule : str
        DESCRIPTION.
    isotopologue : str
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    
    structure = molecule + '(' + isotopologue + ')'
    
    if (molecule in ['MgH', 'CaH']):
        print("\n No single default line list for MgH and CaH. Please specify either 'MoLLIST' or 'Yadin', depending on the desired wavelength range.")
        sys.exit(0)
    
    # Dictionary that defines the default 
    default_list = {'H2(1H2)': 'RACPPK', 'N2(14N2)': 'WCCRMT', 'C2(12C2)': '8states', 'CO(12C-16O)': 'Li2015',
                    'NO(14N-16O)': 'NOname', 'PO(31P-16O)': 'POPS', 'VO(51V-16O)': 'VOMYT', 'YO(89Y-16O)': 'SSYT',
                    'CN(12C-14N)': 'MoLLIST', 'NH(14N-1H)': 'MoLLIST', 'CH(12C-1H)': 'MoLLIST', 'OH(16O-1H)': 'MoLLIST',
                    'SH(32S-1H)': 'GYT', 'HF(1H-19F)': 'Coxon-Hajig', 'CS(12C-32S)': 'JnK', 'NS(14N-32S)': 'SNaSH',
                    'PS(31P-32S)': 'POPS', 'PH(31P-1H)': 'LaTY', 'PN(31P-14N)': 'YYLT', 'CP(12C-31P)': 'MoLLIST',
                    'H2_p(1H-2H_p)': 'ADJSAAM', 'H3_p(1H3_p)': 'MiZATeP', 'OH_p(16O-1H_p)': 'MoLLIST', 
                    'HeH_p(4He-1H_p)': 'ADJSAAM', 'LiH_p(7Li-1H_p)': 'CLT', 'KCl(39K-35Cl)': 'Barton', 
                    'NaCl(23Na-35Cl)': 'Barton', 'LiCl(7Li-35Cl)': 'MoLLIST', 'AlCl(27Al-35Cl)': 'MoLLIST', 
                    'KF(39K-19F)': 'MoLLIST', 'AlF(27Al-19F)': 'MoLLIST', 'LiF(7Li-19F)': 'MoLLIST',
                    'CaF(40Ca-19F)': 'MoLLIST', 'MgF(24Mg-19F)': 'MoLLIST', 'TiO(48Ti-16O)': 'Toto',
                    'TiO(49Ti-16O)': 'Toto', 'TiO(50Ti-16O)': 'Toto', 'TiO(46Ti-16O)': 'Toto', 'TiO(47Ti-16O)': 'Toto',
                    'AlO(27Al-16O)': 'ATP', 'SiO(28Si-16O)': 'EBJT', 'CaO(40Ca-16O)': 'VBATHY', 'MgO(24Mg-16O)': 'LiTY',
                    'NaH(23Na-1H)': 'Rivlin', 'AlH(27Al-1H)': 'AlHambra', 'CrH(52Cr-1H)': 'MoLLIST', 
                    'BeH(9Be-1H)': 'Darby-Lewis', 'TiH(48Ti-1H)': 'MoLLIST', 'FeH(56Fe-1H)': 'MoLLIST', 
                    'LiH(7Li-1H)': 'CLT', 'ScH(45Sc-1H)': 'LYT', 'NaH(23Na-19F)': 'MoLLIST', 'SiH(28Si-1H)': 'SiGHTLY',
                    'SiS(28Si-32S)': 'UCTY', 'H2O(1H2-16O)': 'POKAZATEL', 'HCN(1H-12C-14N)': 'Harris', 
                    'CH4(12C-1H4)': 'YT34to10', 'NH3(14N-1H3)': 'CoYuTe', 'H2S(1H2-32S)': 'AYT2', 
                    'SO2(32S-16O2)': 'ExoAmes', 'SO3(32S-16O3)': 'UYT2', 'PH3(31P-1H3)': 'SAlTY', 'CH3(12C-1H3)': 'AYYJ',
                    'AsH3(75As-1H3)': 'CYT18', 'SiH2(28Si-1H2)': 'CATS', 'SiH4(28Si-1H4)': 'OY2T', 
                    'SiO2(28Si-16O2)': 'OYT3', 'HNO3(1H-14N-16O3)': 'AIJS', 'H2O2(1H2-16O2)': 'APTY', 
                    'H2CO(1H2-12C-16O)': 'AYTY', 'C2H2(12C2-1H2)': 'aCeTY', 'C2H4(12C2-1H4)': 'MaYTY', 
                    'P2H2(31P2-1H2)': 'Trans', 'HPPH(1H-31P2-1H)': 'Cis', 'CH3F(12C-1H3-19F)': 'OYKYT', 
                    'CH3Cl(12C-1H3-35Cl)': 'OYT', 'CO2(12C-16O2)': 'UCL-4000'}
    
    linelist = default_list.get(structure)
    
    if linelist is None:
        print("\n Looks like we haven't specified a default line list to use for this molecule. Try inputting the linelist you want to use as a parameter in the summon() function.")
        sys.exit(0)
    else:
        return linelist


def determine_linelist():
    """
    Determines the desired molecular line list from ExoMol from user input

    Returns
    -------
    molecule : String
        DESCRIPTION.
    isotopologue : String
        DESCRIPTION.
    linelist : String
        DESCRIPTION.
    website : String
        DESCRIPTION.

    """
    
    website = "http://exomol.com/data/molecules/"
    
    while True:
        try:
            molecule = input('What molecule would you like to download the line list for (This is case-sensitive)?\n')
            molecule = re.sub('[+]', '_p', molecule)
            response = requests.get(website + molecule + '/')
            response.raise_for_status() # Raises HTTPError if a bad request is made (server or client error)
            
        except requests.HTTPError:
            print("\n ----- This is not a valid molecule, please try again -----")
        
        else: 
            website += molecule + '/'
            break
        
        
    while True:
        try:
            isotopologue = input("What isotopologue of this molecule would you like to use (type 'default' to use the default isotopologue)?\n")
            if isotopologue.lower() == 'default':
                isotopologue = get_default_iso(molecule)
                website += isotopologue + '/'
                break
            isotopologue = re.sub('[+]', '_p', isotopologue)
            response = requests.get(website + isotopologue + '/')
            response.raise_for_status()
        
        except requests.HTTPError:
            print("\n ----- This is not a valid isotopologue, please try again -----")
        
        else: 
            website += isotopologue + '/'
            break
    
    while True:
        try: 
            linelist = input("Which line list of this isotopologue would you like to use (type 'default' to use the default line list)?\n")
            
            if linelist.lower() == 'default':
                linelist = get_default_linelist(molecule, isotopologue)
                website += linelist + '/'
                return molecule, isotopologue, linelist, website
                break
            
            response = requests.get(website + linelist + '/')
            response.raise_for_status()
    
        except requests.HTTPError:
            print("\n ----- This is not a valid line list, please try again -----")
        
        else:
            website += linelist + '/'
            return molecule, isotopologue, linelist, website
        
        
def process_files(input_dir):
    """
    Processes the .broad and .pf files downloaded from ExoMol into a format that Cthulhu.py can read to create cross-sections

    Parameters
    ----------
    input_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    for file in os.listdir(input_dir):
        if file.endswith('.broad'):
            
            gamma_L_0 = []
            n_L = []
            J = []
        
            in_file_path = input_dir + '/' + file
            f_in = open(in_file_path, 'r')

            for line in f_in:
            
                line = line.strip()
                line = line.split()
    
                if (len(line) == 4):
                
                    gamma_L_0.append(float(line[1]))
                    n_L.append(float(line[2]))
                    J.append(float(line[3]))
                    
            f_in.close()
    
            out_file = './' + file
            f_out = open(out_file, 'w')
    
            f_out.write('J | gamma_L_0 | n_L \n')
    
            for i in range(len(J)):
                f_out.write('%.1f %.4f %.3f \n' %(J[i], gamma_L_0[i], n_L[i]))
        
            f_out.close()
        
            os.remove(in_file_path)
            shutil.move(out_file, input_dir)
            
        
        elif file.endswith('.pf'):
        
            T_pf = []
            Q = []

            in_file_path = input_dir + '/' + file
            f_in = open(in_file_path, 'r')

            for line in f_in:
            
                line = line.strip()
                line = line.split()
                
                T_pf.append(float(line[0]))
                Q.append(float(line[1]))
      
            f_in.close()
    
            out_file = './' + file
            f_out = open(out_file, 'w')
    
            f_out.write('T | Q \n') 
    
            for i in range(len(T_pf)):
                f_out.write('%.1f %.4f \n' %(T_pf[i], Q[i]))
        
            f_out.close()
        
            os.remove(in_file_path)
            shutil.move(out_file, input_dir)
          

def summon_ExoMol(molecule, isotopologue, line_list, URL):
    """
    Main function, uses calls to other functions to perform the download

    Parameters
    ----------
    molecule : TYPE
        DESCRIPTION.
    isotopologue : TYPE
        DESCRIPTION.
    line_list : TYPE
        DESCRIPTION.
    URL : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    line_list_folder = download.create_directories(molecule = molecule, isotopologue = isotopologue, 
                                                   line_list = line_list, database = 'ExoMol')

    host = "http://exomol.com" 
    broad_URL = "http://exomol.com/data/molecules/" + molecule + '/' # URL where the broadening files are contained

    tags = download.create_ExoMol_tag_array(URL, broad_URL)
    
    print("\n ***** Downloading requested data from ExoMol. You have chosen the following parameters: ***** ")
    print("\nMolecule:", molecule, "\nIsotopologue:", isotopologue, "\nLine List:", line_list)
    print("\nStarting by downloading the .broad, .pf, and .states files...")
    
    download.iterate_ExoMol_tags(tags, host, line_list_folder, line_list)
    
    process_files(line_list_folder)
    
    
def load_states(input_directory):
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