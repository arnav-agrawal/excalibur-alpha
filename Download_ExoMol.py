#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:58:01 2020

@author: arnav
"""

from bs4 import BeautifulSoup
import requests
import re
import os
import bz2
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import time
import shutil


def download_file(url, f, l_folder):
    """
    Download a file from ExoMol and decompress it if needed. 
    
    Parameters:
        url (string): The URL of a given ExoMol file
        f (string): The filename of the resulting downloaded file
    """
    
    if f.endswith('bz2') == True: # If the file ends in .bz2 we need to read and decompress it
        
        # Check if the file was already downloaded
        if (os.path.splitext(os.path.splitext(f)[0])[0] + '.h5') in os.listdir(l_folder):
            print("This file is already downloaded. Moving on.")
            return
        
        # Create directory location to prepare for reading compressed file
        compressed_file = l_folder + '/' + f
        
        # Create a decompresser object and a directory location for the decompressed file
        decompressor = bz2.BZ2Decompressor()
        decompressed_file = l_folder + '/' + os.path.splitext(f)[0] #Keep the file name but get rid of the .bz2 extension to make it .trans
        
    
        
        # Download file from the given URL in chunks and then decompress that chunk immediately
        with requests.get(url, stream=True) as request:
            
            if f.find("trans") != -1: # Only want to include the progress bar for .trans downloads
                with open(compressed_file, 'wb') as file, open(decompressed_file, 'wb') as output_file, tqdm(total = int(request.headers.get('content-length', 0)), unit = 'iB', unit_scale = True) as pbar:
                    for chunk in request.iter_content(chunk_size = 1024 * 1024):
                        file.write(chunk)
                        pbar.update(len(chunk))
                        output_file.write(decompressor.decompress(chunk))
                        
                convert_to_hdf(decompressed_file)
                        
            else:
                with open(compressed_file, 'wb') as file, open(decompressed_file, 'wb') as output_file:
                    for chunk in request.iter_content(chunk_size = 1024 * 1024):
                        file.write(chunk)
                        output_file.write(decompressor.decompress(chunk))
                        

        # Delete the compressed file
        os.remove(compressed_file)
                    
                    
    else: # If the file is not compressed we just need to read it in
        
        if 'air' in f:
            input_file = l_folder + '/air.broad'
        elif 'self' in f:
            input_file = l_folder + '/self.broad'
        else:
            input_file = l_folder + '/' + f
        
        with requests.get(url, stream=True) as request:
            with open(input_file, 'wb') as file:
                for chunk in request.iter_content(chunk_size = 1024 * 1024):
                    file.write(chunk)    
    
    
def convert_to_hdf(file):
    """
    Convert a given file to HDF5 format

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    print("Converting this .trans file to HDF to save storage space...")
    
    start_time = time.time()
    
    trans_file = pd.read_csv(file, delim_whitespace = True, header=None, usecols = [0,1,2])
    
    upper_state = np.array(trans_file[0])
    lower_state = np.array(trans_file[1])
    log_Einstein_A = np.log10(np.array(trans_file[2]))   
    
    hdf_file_path = os.path.splitext(file)[0] + '.h5'
    
    with h5py.File(hdf_file_path, 'w') as hdf:
        hdf.create_dataset('Upper State', data = upper_state, dtype = 'u4') #store as 32-bit unsigned int
        hdf.create_dataset('Lower State', data = lower_state, dtype = 'u4') #store as 32-bit unsigned int
        hdf.create_dataset('Log Einstein A', data = log_Einstein_A, dtype = 'f4') #store as 32-bit float

    os.remove(file)
    
    print("This .trans file took", round(time.time() - start_time, 1), "seconds to reformat to HDF.")
    


def create_tag_array(url, broad_URL):
    """
    Create a list of html tags that contain the URLs from which we will later download files
    
    Parameters:
        url (string): The ExoMol URL for the webpage that contains download links to all the files
    """
    
    # Get webpage content as text
    web_content = requests.get(url).text
    broadening_content = requests.get(broad_URL).text
    
    # Create lxml parser
    soup = BeautifulSoup(web_content, "lxml")
    soup2 = BeautifulSoup(broadening_content, "lxml")

    # Parse the webpage by file type (which is contained in the href of the html tag)
    broad_tags = soup2.find_all('a', href = re.compile("broad"))
    pf_tags = soup.find_all('a', href = re.compile("pf"))
    states_tags = soup.find_all('a', href = re.compile("states"))
    trans_tags = soup.find_all('a', href = re.compile("trans"))

    combined_tags = broad_tags + pf_tags + states_tags + trans_tags
    
    return combined_tags


def create_directories(molecule, isotope, line_list):
    """
    Create new folders to store the rdesired data

    Parameters
    ----------
    molecule : TYPE
        DESCRIPTION.
    isotope : TYPE
        DESCRIPTION.
    line_list : TYPE
        DESCRIPTION.

    Returns
    -------
    molecule_folder : TYPE
        DESCRIPTION.
    line_list_folder : TYPE
        DESCRIPTION.

    """
    
    input_folder = '../input'
    molecule_folder = input_folder + '/' + molecule + '  ~  (' + isotope + ')'
    line_list_folder = molecule_folder + '/' + line_list
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if os.path.exists(molecule_folder) == False:
        os.mkdir(molecule_folder)

    if os.path.exists(line_list_folder) == False:
        os.mkdir(line_list_folder)
        
    return line_list_folder

        

def calc_num_trans(html_tags):
    """
    Calculate the number of .trans files in the line list

    Parameters
    ----------
    html_tags : TYPE
        DESCRIPTION.

    Returns
    -------
    counter : TYPE
        DESCRIPTION.

    """
    
    counter = 0
    for tag in html_tags:
        if tag.get('href').find('trans') != -1:
            counter += 1
    return counter



def iterate_tags(tags, host, l_folder, line_list):
    """
    Iterate through every html tag and download the file contained by the URL in the href

    Parameters
    ----------
    tags : TYPE
        DESCRIPTION.
    host : TYPE
        DESCRIPTION.
    l_folder : TYPE
        DESCRIPTION.
    line_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    counter = 0
    num_trans = calc_num_trans(tags)
    
    for tag in tags:
        # Create the appropriate URL by combining host name and href
        url = host + tag.get('href')
         
        # Name the file in a way that it includes relevant info about what is stored in the file
        matches = re.finditer('__', url)
        matches_positions = [match.start() for match in matches]
        filename = url[matches_positions[len(matches_positions) - 1] + 2:]
        
        if filename.find('trans') != - 1:
            counter += 1
            if counter == 1:
                print("Fetched the broadening coefficients, partition functions, and energy levels.")
                print("Now downloading the", line_list, "line list...")
            
            print("\nDownloading .trans file", counter, "of", num_trans)
        
        # Download each line list
        download_file(url, filename, l_folder)
        
        

def define_url(molecule, isotope, line_list):
    """ Generate an ExoMol URL from which various files will be downloaded, given a molecule, isotope, and line_list
    
    """
        
    ExoMol_URL = 'http://exomol.com/data/molecules/' + molecule + '/' + isotope + '/' + line_list + '/'
    
    return ExoMol_URL


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
    
    #HCl(1H-35Cl): HITRAN2016    <-- data from HITRAN, but in ExoMol format
    #MgH(24Mg-1H): MoLLIST / Yadin    <-- we need to check, each line list seems to cover a different wavelength range
    #CaH (40Ca-1H): MoLLIST / Yadin    <-- we need to check, each line list seems to cover a different wavelength range 
    
    structure = molecule + '(' + isotopologue + ')'
    
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
    
    return default_list.get(structure)



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
    
    line_list_folder = create_directories(molecule, isotopologue, line_list)

    host = "http://exomol.com" 
    broad_URL = "http://exomol.com/data/molecules/" + molecule + '/' # URL where the broadening files are contained

    tags = create_tag_array(URL, broad_URL)
    
    print("\n ***** Downloading requested data from ExoMol. You have chosen the following parameters: ***** ")
    print("\nMolecule:", molecule, "\nIsotopologue:", isotopologue, "\nLine List:", line_list)
    print("\nStarting by downloading the .broad, .pf, and .states files...")
    
    iterate_tags(tags, host, line_list_folder, line_list)
    
    process_files(line_list_folder)
    
    
