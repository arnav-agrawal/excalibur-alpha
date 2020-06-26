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
import numpy
import pandas
import h5py
import time


def download_file(url, f, m_folder, l_folder):
    """
    Download a file from ExoMol - decompress if needed
    
    Parameters:
        url (string): The URL of a given ExoMol file
        f (string): The filename of the resulting downloaded file
    """
    
    if f.endswith('bz2') == True: # If the file ends in .bz2 we need to read and decompress it
        
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
                        
                #convert_to_hdf(decompressed_file)
                        
            else:
                with open(compressed_file, 'wb') as file, open(decompressed_file, 'wb') as output_file:
                    for chunk in request.iter_content(chunk_size = 1024 * 1024):
                        file.write(chunk)
                        output_file.write(decompressor.decompress(chunk))
                        

        # Delete the compressed file
        os.remove(compressed_file)
                    
                    
    else: # If the file is not compressed we just need to read it in
        
        if f.endswith('broad') == True: # Include this condition because .broad files are placed in a separate directory location
            input_file = l_folder + '/' + f
            #print("Reading this file from ExoMol:", input_file)
            with requests.get(url, stream=True) as request:
                with open(input_file, 'wb') as file:
                    for chunk in request.iter_content(chunk_size = 1024 * 1024):
                        file.write(chunk)
                        
        else: # the file must be .pf
            input_file = l_folder + '/' + f
            #print("Reading this file from ExoMol:", input_file)
            with requests.get(url, stream=True) as request:
                with open(input_file, 'wb') as file:
                    for chunk in request.iter_content(chunk_size = 1024 * 1024):
                        file.write(chunk)
    
    
    
def convert_to_hdf(file):
    """ Convert a given file to HDF5 format. Used for the .trans files.
    
    :param file: The .trans file that will be converted to .hdf format
    :type file: String
    """
    
    print("Converting this .trans file to HDF to save storage space...")
    
    start_time = time.time()
    
    trans_file = pandas.read_csv(file, delim_whitespace = True, header=None, usecols = [0,1,2])
    
    upper_state = numpy.array(trans_file[0])
    lower_state = numpy.array(trans_file[1])
    log_Einstein_A = numpy.log10(numpy.array(trans_file[2]))   
    
    hdf_file_path = os.path.splitext(file)[0] + '.h5'
    
    with h5py.File(hdf_file_path, 'w') as hdf:
        hdf.create_dataset('Upper State', data = upper_state, dtype = 'u4') #store as 32-bit unsigned int
        hdf.create_dataset('Lower State', data = lower_state, dtype = 'u4') #store as 32-bit unsigned int
        hdf.create_dataset('Log Einstein A', data = log_Einstein_A, dtype = 'f4') #store as 32-bit float

    os.remove(file)
    
    print("This .trans file took", round(time.time() - start_time, 1), "seconds to reformat to HDF.")
    


def create_tag_array(url):
    """
    Create a list of html tags that contain the URLs from which we will later download files
    
    Parameters:
        url (string): The ExoMol URL for the webpage that contains download links to all the files
    """
    
    # Get webpage content as text
    web_content = requests.get(url).text
    
    # Create lxml parser
    soup = BeautifulSoup(web_content, "lxml")

    # Parse the webpage by file type (which is contained in the href of the html tag)
    broad_tags = soup.find_all('a', href = re.compile("broad"))
    pf_tags = soup.find_all('a', href = re.compile("pf"))
    states_tags = soup.find_all('a', href = re.compile("states"))
    trans_tags = soup.find_all('a', href = re.compile("trans"))

    combined_tags = broad_tags + pf_tags + states_tags + trans_tags
    
    return combined_tags


def create_directories(molecule, isotope, line_list):
    """ Create new folders to store the relevant data
    
    :return: The names of directories that have been created
    :rtype: tuple
    """
    
    input_folder = '../input'
    molecule_folder = input_folder + '/' + molecule + '(' + isotope + ')'
    line_list_folder = molecule_folder + '/' + line_list
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if os.path.exists(molecule_folder) == False:
        os.mkdir(molecule_folder)

    if os.path.exists(line_list_folder) == False:
        os.mkdir(line_list_folder)
        
    return molecule_folder, line_list_folder

        

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



def iterate_tags(tags, host, m_folder, l_folder, line_list):
    """ Iterate through every html tag and download the file contained by the URL in the href
    
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
        download_file(url, filename, m_folder, l_folder)
        
        

def define_url(molecule, isotope, line_list):
    """ Generate an ExoMol URL from which various files will be downloaded, given a molecule, isotope, and line_list
    
    """
        
    ExoMol_URL = 'http://exomol.com/data/molecules/' + molecule + '/' + isotope + '/' + line_list + '/'
    
    #create_directories(molecule, isotope, line_list)
    
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
    matches = re.findall('[A-Z][a-z]?[0-9]?', molecule)
    num_matches = len(matches)
    for match in matches:
        num_matches -= 1
        letters = re.findall('\D+', match) # alphabetic characters in the molecule
        numbers = re.findall('\d', match) # numeric characters in the molecule
        default_iso += str(most_abundant.get(letters[0])) + letters[0] 
        
        if numbers:
            default_iso += numbers[0]
            
        if num_matches != 0:
            default_iso += '-'
        
    return default_iso


def get_default_linelist(molecule, isotopologue):
    return


def summon_ExoMol(molecule, isotopologue, line_list, URL):
    """ Main function, called by the user to start the download process
    
    """
    
    (molecule_folder, line_list_folder) = create_directories(molecule, isotopologue, line_list)

    host = "http://exomol.com" 

    tags = create_tag_array(URL)
    
    print("\n ***** Downloading requested data from ExoMol. You have chosen the following parameters: ***** ")
    print("\nMolecule:", molecule, "\nIsotopologue:", isotopologue, "\nLine List:", line_list)
    print("\nStarting by downloading the .broad, .pf, and .states files...")
    
    iterate_tags(tags, host, molecule_folder, line_list_folder, line_list)
    
    return line_list_folder
    
    
##### Begin Main Program #####   
