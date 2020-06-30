#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:08:06 2020

@author: arnav
"""

from bs4 import BeautifulSoup
import requests
import pandas
import os
import time
import numpy
import bz2
import h5py
from tqdm import tqdm
from hapi import moleculeName, isotopologueName, abundance

def HITEMP_table():
    url = 'https://hitran.org/hitemp/'

    web_content = requests.get(url).text
    
    soup = BeautifulSoup(web_content, "lxml")
    table = soup.find('table')

    n_rows = 0
    n_columns = 0
    column_names = []

    for row in table.find_all('tr'):
        td_tags = row.find_all('td')
        if len(td_tags) > 0:
            
            n_rows += 1
            
        if n_columns == 0: # Use the number of td tags in the first row to set the first column
            n_columns = len(td_tags)
            
            # Handle column names
            th_tags = row.find_all('th') 
            
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())
            
         
    hitemp = pandas.DataFrame(columns = column_names, index= range(0,n_rows))

    row_marker = 0
    for row in table.find_all('tr'):
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            hitemp.iat[row_marker,column_marker] = column.get_text()
            column_marker += 1
        
        if len(columns) > 0:
            row_marker += 1
        

    hitemp = hitemp[:-1]
    hitemp.rename(columns = {'Iso counta':'Iso Count'}, inplace = True)
    hitemp.loc[len(hitemp)] = ['4', 'N2O', 'Nitrous Oxide', '5', '3626425', '0', '12899', '2019', '']
    hitemp.loc[:, 'ID'] = pandas.to_numeric(hitemp['ID'])
    hitemp.loc[:, 'Iso Count'] = pandas.to_numeric(hitemp['Iso Count'])


    hitemp.sort_values(by = 'ID', inplace = True)
    hitemp.reset_index(drop = True, inplace = True)


    counter = 0
    for tag in table.find_all('a'):
        hitemp.loc[counter, 'Download'] = 'https://hitran.org' + tag.get('href')
        counter += 1
        
    return hitemp


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
    line_list_folder = molecule_folder + '/HITEMP'
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if os.path.exists(molecule_folder) == False:
        os.mkdir(molecule_folder)

    if os.path.exists(line_list_folder) == False:
        os.mkdir(line_list_folder)
        
    return line_list_folder


def convert_to_hdf(file):
    """ Convert a given file to HDF5 format. Used for the .trans files.
    
    :param file: The .trans file that will be converted to .hdf format
    :type file: String
    """
    
    print("Converting this .par file to HDF to save storage space...")
    
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


def download_line_list(mol_ID, iso_ID, out_folder):
    table = HITEMP_table()
    row = table.loc[table['ID'] == mol_ID]
    
    download_link = row.loc[row.index.values[0], 'Download']
    
    if download_link.endswith('.bz2'):
        
        # Create directory location to prepare for reading compressed file
        compressed_file = out_folder + '/HITEMP.par.bz2'
        
        # Create a decompresser object and a directory location for the decompressed file
        decompressor = bz2.BZ2Decompressor()
        decompressed_file = out_folder + '/HITEMP.par' #Keep the file name but get rid of the .bz2 extension to make it .par
    
        
        # Download file from the given URL in chunks and then decompress that chunk immediately
        with requests.get(download_link, stream=True) as request:
            with open(compressed_file, 'wb') as file, open(decompressed_file, 'wb') as output_file, tqdm(total = int(request.headers.get('content-length', 0)), unit = 'iB', unit_scale = True) as pbar:
                for chunk in request.iter_content(chunk_size = 1024 * 1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
                    output_file.write(decompressor.decompress(chunk))
                        
            #convert_to_hdf(decompressed_file)
                        

        # Delete the compressed file
        os.remove(compressed_file)    
        


def summon_HITEMP(molecule, isotopologue):
    output_folder = create_directories(molecule, isotopologue)
    download_line_list(molecule, isotopologue, output_folder)
    
    return output_folder, abundance(molecule, isotopologue)  #Used for processing the downloaded .par file later

"""
table = HITEMP_table()
link = table.loc[4, 'Download']
print(link)

request = requests.get(link)
print(request.headers.get('content-length', 0))
"""