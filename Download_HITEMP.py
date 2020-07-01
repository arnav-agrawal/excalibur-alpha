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
import zipfile
import re
import pandas
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


def convert_to_hdf(mol_ID, iso_ID, file):
    """ Convert a given file to HDF5 format. Used for the .trans files.
    
    :param file: The .trans file that will be converted to .hdf format
    :type file: String
    """
    
    print("Converting this .par file to HDF to save storage space...")
    
    start_time = time.time()
    
    field_lengths = [2,1,12,10,10,5,5,10,4,8,2,2,2,2,2,2,2,1,3,2,2,2,2,2,2,2,2,1,3,12,1,3,1,25,6,1,6]
    
    trans_file = pandas.read_fwf(file, widths=field_lengths, header=None)
    
    # Get only the necessary columns from the .par file
    nu_0 = numpy.array(trans_file[2])
    S_ref = numpy.array(trans_file[3]) / abundance(mol_ID, iso_ID)
    gamma_L_0_air = numpy.array(trans_file[5]) / 1.01325   # Convert from cm^-1 / atm -> cm^-1 / bar
    E_lower = numpy.array(trans_file[7])
    n_L_air = numpy.array(trans_file[8])
    J_lower = numpy.array(trans_file[31]) 
    
    hdf_file_path = os.path.splitext(file)[0] + '.h5'
    
    with h5py.File(hdf_file_path, 'w') as hdf:
        hdf.create_dataset('Transition Wavenumber', data = nu_0, dtype = 'u4') #store as 32-bit unsigned int
        hdf.create_dataset('Line Intensity', data = S_ref, dtype = 'u4') #store as 32-bit unsigned int
        hdf.create_dataset('Lower State E', data = E_lower, dtype = 'f4') #store as 32-bit float
        hdf.create_dataset('Lower State J', data = J_lower, dtype = 'f4') #store as 32-bit float

    os.remove(file)
    
    print("This .trans file took", round(time.time() - start_time, 1), "seconds to reformat to HDF.")
    
    return J_lower, gamma_L_0_air, n_L_air



def create_air_broad(J_lower_all, gamma_air, n_air, gamma_air_avg, n_air_avg, input_dir):
    J_max = max(J_lower_all)
    J_sorted = numpy.arange(int(J_max))
    
    for i in range(int(J_max)):
        
        gamma_air_i = numpy.mean(gamma_air[numpy.where(J_lower_all == J_sorted[i])])
        n_air_i = numpy.mean(n_air[numpy.where(J_lower_all == J_sorted[i])])
        
        gamma_air_avg = numpy.append(gamma_air_avg, gamma_air_i)
        n_air_avg = numpy.append(n_air_avg, n_air_i)
    
    # Write air broadening file
    out_file = input_dir + '/air.broad'
    f_out = open(out_file,'w')
    
    f_out.write('J | gamma_L_0 | n_L \n')
    
    for i in range(len(J_sorted)):
        f_out.write('%.1f %.4f %.3f \n' %(J_sorted[i], gamma_air_avg[i], n_air_avg[i]))
        
    f_out.close()



def download_line_list(mol_ID, iso_ID, out_folder):
    print(out_folder)
    table = HITEMP_table()
    row = table.loc[table['ID'] == mol_ID]
    
    download_link = row.loc[row.index.values[0], 'Download']
    
    if download_link.endswith('.bz2'):
        
        # Create directory location to prepare for reading compressed file
        compressed_file = out_folder + '/' + moleculeName(mol_ID) + '.par.bz2'
        
        # Create a decompresser object and a directory location for the decompressed file
        decompressor = bz2.BZ2Decompressor()
        decompressed_file = out_folder + '/' + moleculeName(mol_ID) + '.par' #Keep the file name but get rid of the .bz2 extension to make it .par
    
        
        # Download file from the given URL in chunks and then decompress that chunk immediately
        with requests.get(download_link, stream=True) as request:
            with open(compressed_file, 'wb') as file, open(decompressed_file, 'wb') as output_file, tqdm(total = int(request.headers.get('content-length', 0)), unit = 'iB', unit_scale = True) as pbar:
                for chunk in request.iter_content(chunk_size = 1024 * 1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
                    output_file.write(decompressor.decompress(chunk))
                     
                  
        # Instantiate arrays which will be needed for creating air broadening file
        J_lower_all, gamma_air, n_air = (numpy.array([]) for _ in range(3))
        gamma_air_avg, n_air_avg = (numpy.array([]) for _ in range(2))
        
        # Convert line list to hdf5 file format
        J_i, gamma_air_i, n_air_i = convert_to_hdf(mol_ID, iso_ID, decompressed_file)
        
        # Populate the arrays
        J_lower_all = numpy.append(J_lower_all, J_i)
        gamma_air = numpy.append(gamma_air, gamma_air_i)
        n_air = numpy.append(n_air, n_air_i)
        
        # Create the air broadening file
        create_air_broad(J_lower_all, gamma_air, n_air, gamma_air_avg, n_air_avg, out_folder)

        # Delete the compressed file
        os.remove(compressed_file)    
        
    else:
        new_url = download_link
        web_content = requests.get(new_url).text
    
        soup = BeautifulSoup(web_content, "lxml")
        
        links = []
        fnames = []
        for a in soup.find_all('a'):
            if a.get('href').endswith('.zip'):
                links.append(new_url + a.get('href'))
                fnames.append(a.get('href'))
        
        num_links = len(links)
        counter = 0
        
        for link in links:
            print("\nDownloading .zip file", counter + 1, "of", num_links)
            
            # Create directory location to prepare for reading compressed file
            
            fname = fnames[counter]
            
            compressed_file = out_folder + '/' + fname
            
            with requests.get(link, stream = True) as request:
                with open(compressed_file, 'wb') as file, tqdm(total = int(request.headers.get('content-length', 0)), unit = 'iB', unit_scale = True) as pbar:
                    for chunk in request.iter_content(chunk_size = 1024 * 1024):
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            
            with zipfile.ZipFile(compressed_file, 'r', allowZip64 = True) as file:
                print("Decompressing this file...")
                file.extractall(out_folder + '/')
                
                # Instantiate arrays which will be needed for creating air broadening file
                J_lower_all, gamma_air, n_air = (numpy.array([]) for _ in range(3))
                gamma_air_avg, n_air_avg = (numpy.array([]) for _ in range(2))
                
                # Convert line list to hdf5 file format
                J_i, gamma_air_i, n_air_i = convert_to_hdf(mol_ID, iso_ID, unzipped_file)
                
                # Populate the arrays
                J_lower_all = numpy.append(J_lower_all, J_i)
                gamma_air = numpy.append(gamma_air, gamma_air_i)
                n_air = numpy.append(n_air, n_air_i)
                
                # Create the air broadening file
                create_air_broad(J_lower_all, gamma_air, n_air, gamma_air_avg, n_air_avg, out_folder)
                
            counter += 1
            
            os.remove(compressed_file)
                
        

def summon_HITEMP(molecule, isotopologue):
    output_folder = create_directories(molecule, isotopologue)
    download_line_list(molecule, isotopologue, output_folder)
    

"""
table = HITEMP_table()
link = table.loc[4, 'Download']
print(link)

request = requests.get(link)
print(request.headers.get('content-length', 0))
"""