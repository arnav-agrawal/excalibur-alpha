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
from hapi import moleculeName, isotopologueName, abundance

def HITEMP_table():
    """
    Recreate the table found on the HITEMP main page in order to simplify later processes

    Returns
    -------
    hitemp : TYPE
        DESCRIPTION.

    """
    
    url = 'https://hitran.org/hitemp/'

    web_content = requests.get(url).text
    
    soup = BeautifulSoup(web_content, "lxml")
    table = soup.find('table')

    n_rows = 0
    n_columns = 0
    column_names = []

    # iterate through 'tr' (table row) tags
    for row in table.find_all('tr'):
        
        td_tags = row.find_all('td') # find all 'td' (table data) tags
        
        if len(td_tags) > 0:
            n_rows += 1
            
        if n_columns == 0: # Use the number of td tags in the first row to set the first column
            n_columns = len(td_tags)
            
            # Handle column names
            th_tags = row.find_all('th') 
            
        if len(th_tags) > 0 and len(column_names) == 0: # This loop adds all column headers of the table to 'column_names'
            for th in th_tags:
                column_names.append(th.get_text())
            
         
    hitemp = pandas.DataFrame(columns = column_names, index= range(0,n_rows)) # Create a DataFrame to store the table

    row_marker = 0
    for row in table.find_all('tr'): # This loop populates our DataFrame with the same data as the table online
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            hitemp.iat[row_marker,column_marker] = column.get_text()
            column_marker += 1
        
        if len(columns) > 0:
            row_marker += 1
        

    hitemp = hitemp[:-1] # Get rid of last row
    hitemp.rename(columns = {'Iso counta':'Iso Count'}, inplace = True) # Rename a column header
    hitemp.loc[len(hitemp)] = ['4', 'N2O', 'Nitrous Oxide', '5', '3626425', '0', '12899', '2019', ''] # Manually add N2O molecule to table
    hitemp.loc[:, 'ID'] = pandas.to_numeric(hitemp['ID'])  # This line and the next convert all values in 'ID' and 'Iso Count' column to floats
    hitemp.loc[:, 'Iso Count'] = pandas.to_numeric(hitemp['Iso Count'])


    hitemp.sort_values(by = 'ID', inplace = True)
    hitemp.reset_index(drop = True, inplace = True)


    counter = 0
    for tag in table.find_all('a'): # Populate the 'Download' columns with the url links to the line lists
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
    line_list_folder = molecule_folder + '/HITEMP/'
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if os.path.exists(molecule_folder) == False:
        os.mkdir(molecule_folder)

    if os.path.exists(line_list_folder) == False:
        os.mkdir(line_list_folder)
        
    return line_list_folder


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
        
    if mol_ID == 6 and iso_ID in {1, 2} or mol_ID == 30:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 2, 3, 2, 3, 5, 6, 12, 1, 7, 7]
        J_col = 14
        
    if mol_ID in {11, 24, 27, 28, 39} or mol_ID == 6 and iso_ID in {3, 4}:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 2, 2, 1, 4, 6, 12, 1, 7, 7]
        J_col = 13
        
    if mol_ID == 7:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 1, 1, 3, 1, 3, 5, 1, 6, 12, 1, 7, 7]
        J_col = 17
        
    if mol_ID in {8, 18}:
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 1, 5, 1, 5, 6, 12, 1, 7, 7]
        J_col = 15
        
    if mol_ID in {13}: #Handle OH molecule since it has slightly different format
        field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 1, 5, 1, 5, 6, 12, 1, 7, 7]
        J_col = 15
        
    
    trans_file = pandas.read_fwf(file, widths=field_lengths, header=None)
    trans_file = trans_file.query('@trans_file[1] == @iso_ID') # filter by the requested isotopologue ID
    
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
    input_dir : str
        Location of the transition files used to create the 'air.broad' file.

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
        f_out.write('%.1f %.4f %.3f \n' %(J_sorted[i], gamma_air_avg[i], n_air_avg[i]))
    
    f_out.close()



def download_line_list(mol_ID, iso_ID, out_folder):
    """
    Download a line list(s) from the HITEMP database

    Parameters
    ----------
    mol_ID : TYPE
        DESCRIPTION.
    iso_ID : TYPE
        DESCRIPTION.
    out_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    table = HITEMP_table()
    row = table.loc[table['ID'] == mol_ID]
    
    download_link = row.loc[row.index.values[0], 'Download']
    
    if download_link.endswith('.bz2'):
        
        # Create directory location to prepare for reading compressed file
        compressed_file = out_folder + moleculeName(mol_ID) + '.par.bz2'
        
        # Create a decompresser object and a directory location for the decompressed file
        decompressor = bz2.BZ2Decompressor()
        decompressed_file = out_folder + moleculeName(mol_ID) + '.par' #Keep the file name but get rid of the .bz2 extension to make it .par
    
        
        # Download file from the given URL in chunks and then decompress that chunk immediately
        with requests.get(download_link, stream=True) as request:
            with open(compressed_file, 'wb') as file, open(decompressed_file, 'wb') as output_file, tqdm(total = int(request.headers.get('content-length', 0)), unit = 'iB', unit_scale = True) as pbar:
                for chunk in request.iter_content(chunk_size = 1024 * 1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
                    output_file.write(decompressor.decompress(chunk))
                     
        
        # Convert line list to hdf5 file format
        print("Converting this .par file to HDF to save storage space...")
        convert_to_hdf(mol_ID, iso_ID, decompressed_file)

        # Delete the compressed file
        os.remove(compressed_file)    
        
    else:
        # 'download_link' will take us to another site containing one or more '.zip' files that need to be downlaoded
        new_url = download_link
        web_content = requests.get(new_url).text
    
        soup = BeautifulSoup(web_content, "lxml")
        
        links = []
        fnames = []
        for a in soup.find_all('a'):  # Parse the webpage to find all 'href' tags that end with '.zip'
            if a.get('href').endswith('.zip'):
                links.append(new_url + a.get('href'))
                fnames.append(a.get('href'))
        
        num_links = len(links)
        counter = 0
        
        for link in links:
            print("\nDownloading .zip file", counter + 1, "of", num_links)
    
            fname = fnames[counter]
            compressed_file = out_folder + fname
            
            # Download compressed version of file
            with requests.get(link, stream = True) as request:
                with open(compressed_file, 'wb') as file, tqdm(total = int(request.headers.get('content-length', 0)), unit = 'iB', unit_scale = True) as pbar:
                    for chunk in request.iter_content(chunk_size = 1024 * 1024):
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            
            # Decompress the file
            with zipfile.ZipFile(compressed_file, 'r', allowZip64 = True) as file:
                print("Decompressing this file...")
                file.extractall(out_folder)
                
            counter += 1
            os.remove(compressed_file)
            
        counter = 0    
        for file in os.listdir(out_folder): # Convert all downloaded files to HDF5
            if file.endswith('.par'):
                print("\nConverting .par file", counter + 1, "of", num_links, "to HDF to save storage space.")
                convert_to_hdf(mol_ID, iso_ID, out_folder + file)
                counter += 1
        

def summon_HITEMP(molecule, isotopologue):
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
    print("\nFetching data from HITEMP...\nMolecule:", moleculeName(molecule), "\nIsotopologue", isotopologueName(molecule, isotopologue), "\n")
    
    output_folder = create_directories(molecule, isotopologue)
    download_line_list(molecule, isotopologue, output_folder)
    print("\n\nNow creating air broadening file ...")
    create_air_broad(output_folder)
    
    return output_folder
