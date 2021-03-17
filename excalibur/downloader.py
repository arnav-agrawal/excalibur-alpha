
from bs4 import BeautifulSoup
import requests
import re
import os
import bz2
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import time
import shutil
import zipfile

from .hapi import db_begin, fetch, abundance, moleculeName, isotopologueName

import excalibur.ExoMol as ExoMol
import excalibur.HITRAN as HITRAN

def download_ExoMol_file(url, f, l_folder):
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
                
                print("Converting this .trans file to HDF to save storage space...")
                convert_to_hdf(file = decompressed_file, database = 'ExoMol')
                        
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
    
    
def download_HITRAN_line_list(mol_ID, iso_ID, folder, nu_min = 1, nu_max = 100000):
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
            
         
    hitemp = pd.DataFrame(columns = column_names, index= range(0,n_rows)) # Create a DataFrame to store the table

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
    hitemp.loc[:, 'ID'] = pd.to_numeric(hitemp['ID'])  # This line and the next convert all values in 'ID' and 'Iso Count' column to floats
    hitemp.loc[:, 'Iso Count'] = pd.to_numeric(hitemp['Iso Count'])


    hitemp.sort_values(by = 'ID', inplace = True)
    hitemp.reset_index(drop = True, inplace = True)


    counter = 0
    for tag in table.find_all('a'): # Populate the 'Download' columns with the url links to the line lists
        hitemp.loc[counter, 'Download'] = 'https://hitran.org' + tag.get('href')
        counter += 1
        
    return hitemp


def download_HITEMP_line_list(mol_ID, iso_ID, out_folder):
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
        convert_to_hdf(file = decompressed_file, mol_ID = mol_ID,
                       iso_ID = iso_ID, database = 'HITEMP')

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
                convert_to_hdf(file = (out_folder + file), mol_ID = mol_ID, 
                               iso_ID = iso_ID, database = 'HITEMP')
                counter += 1
        
    
def convert_to_hdf(file = '', mol_ID = '', iso_ID = '', alkali = False, 
                   database = '', **kwargs):
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
    
    start_time = time.time()
    
    if (database == 'ExoMol'):
    
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
        
    elif (database in ['HITRAN', 'HITEMP']):
        
        # Different HITRAN formats for different molecules leads us to read in .par files w/ different field widths
        if mol_ID in {1, 3, 9, 12, 20, 21, 25, 29, 31, 32, 35, 37, 38}:  # Group 1
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 3, 5, 1, 6, 12, 1, 7, 7]
            J_col = 13
            
        elif mol_ID in {10, 33}:  # Group 1 - Handle HO2 and NO2 J_cols seperately, since HITRAN provides N instead of J
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 3, 5, 1, 6, 12, 1, 7, 7]
            J_col = 13
            Sym_col = 17
        
        elif mol_ID in {2, 4, 5, 14, 15, 16, 17, 19, 22, 23, 26, 36}:  # Group 2
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 5, 1, 3, 1, 5, 6, 12, 1, 7, 7]
            J_col = 15
            
        elif (mol_ID == 6 and iso_ID in {1, 2}) or mol_ID == 30:  # Group 3
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 2, 3, 2, 3, 5, 6, 12, 1, 7, 7]
            J_col = 14
            
        elif mol_ID in {11, 24, 27, 28, 39} or (mol_ID == 6 and iso_ID in {3, 4}):  # Group 4
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 3, 2, 2, 1, 4, 6, 12, 1, 7, 7]
            J_col = 13
            
        elif mol_ID == 7:  # Group 5
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 1, 1, 3, 1, 3, 5, 1, 6, 12, 1, 7, 7]
            J_col = 17
            
        elif mol_ID in {8, 18}:  # Group 6
            field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 1, 5, 1, 5, 6, 12, 1, 7, 7]
            J_col = 15
        
        # OH has slightly different field widths between HITRAN and HITEMP
        if (database == 'HITRAN'):
            if mol_ID in {13}: # Group 6 - OH
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 5, 2, 5, 6, 12, 1, 7, 7]
                J_col = 14
        elif (database == 'HITEMP'):
            if mol_ID in {13}: # Group 6 - OH
                field_lengths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 3, 1, 5, 1, 5, 6, 12, 1, 7, 7]
                J_col = 15
        
        trans_file = pd.read_fwf(file, widths=field_lengths, header=None)
            
        if (database == 'HITEMP'):
            trans_file = trans_file.query('@trans_file[1] == @iso_ID') # filter by the requested isotopologue ID
        
        # Get only the necessary columns from the .par file
        nu_0 = np.array(trans_file[2])
        log_S_ref = np.log10(np.array(trans_file[3]) / abundance(mol_ID, iso_ID))
        gamma_L_0_air = np.array(trans_file[5]) / 1.01325   # Convert from cm^-1 / atm -> cm^-1 / bar
        E_lower = np.array(trans_file[7])
        n_L_air = np.array(trans_file[8])
        J_lower = np.array(trans_file[J_col])
        
        if mol_ID in {10, 33}:  # Handle creation of NO2 and HO2 J_lower columns, as the given value is N on HITRAN not J
            Sym = np.array(trans_file[Sym_col])
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

    elif (database == 'VALD'):
        
        trans_file = pd.read_csv(file, delim_whitespace = True, header=None, skiprows = 1)
        
        nu_0 = np.array(trans_file[0])
        log_gf = np.array(trans_file[1])
        E_low = np.array(trans_file[2])
        E_up = np.array(trans_file[3])
        J_low = np.array(trans_file[4])
        J_up = np.array(trans_file[5])
        
        # Account for the differences in columns depending on if the species is an alkali metal
        if (alkali == True):  
            l_low = np.array(trans_file[6])
            l_up = np.array(trans_file[7])
            log_gamma_nat = np.array(trans_file[8])
            log_gamma_vdw = np.array(trans_file[9])
            
        else:
            log_gamma_nat = np.array(trans_file[6])
            log_gamma_vdw = np.array(trans_file[7])
            
        hdf_file_path = os.path.splitext(file)[0] + '.h5'
        
        with h5py.File(hdf_file_path, 'w') as hdf:
            hdf.create_dataset('nu', data = nu_0, dtype = 'f8') #store as 32-bit unsigned int
            hdf.create_dataset('Log gf', data = log_gf, dtype = 'f8') #store as 32-bit unsigned int
            hdf.create_dataset('E lower', data = E_low, dtype = 'f8') #store as 32-bit float
            hdf.create_dataset('E upper', data = E_up, dtype = 'f8') #store as 32-bit unsigned int
            hdf.create_dataset('J lower', data = J_low, dtype = 'f8') #store as 32-bit unsigned int
            hdf.create_dataset('J upper', data = J_up, dtype = 'f8') #store as 32-bit float
            hdf.create_dataset('Log gamma nat', data = log_gamma_nat, dtype = 'f8') #store as 32-bit float
            hdf.create_dataset('Log gamma vdw', data = log_gamma_vdw, dtype = 'f8') #store as 32-bit float
            
            if (alkali == True):
                hdf.create_dataset('l lower', data = l_low, dtype = 'f8') #store as 32-bit unsigned int
                hdf.create_dataset('l upper', data = l_up, dtype = 'f8') #store as 32-bit float

    
    print("This .trans file took", round(time.time() - start_time, 1), "seconds to reformat to HDF.")
    

def create_directories(molecule = '', isotopologue = '', line_list = '', database = '',
                       mol_ID = '', iso_ID = '', ionization_state = 1, VALD_data_dir = '',
                       **kwargs):
    '''
    Create new folders to store the relevant data

    Parameters
    ----------
    molecule : TYPE, optional
        DESCRIPTION. The default is ''.
    isotope : TYPE, optional
        DESCRIPTION. The default is ''.
    line_list : TYPE, optional
        DESCRIPTION. The default is ''.
    mol_ID : TYPE, optional
        DESCRIPTION. The default is ''.
    iso_ID : TYPE, optional
        DESCRIPTION. The default is ''.
    ionization_state : TYPE, optional
        DESCRIPTION. The default is 1.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    line_list_folder : TYPE
        DESCRIPTION.

    '''
    
    input_folder = './input'
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if (database == 'ExoMol'):
        
        molecule_folder = input_folder + '/' + molecule + '  ~  (' + isotopologue + ')'
        database_folder = molecule_folder + '/' + database + '/'
        line_list_folder = molecule_folder + '/' + database + '/' + line_list + '/'
        
        if os.path.exists(molecule_folder) == False:
            os.mkdir(molecule_folder)
            
        if os.path.exists(database_folder) == False:
            os.mkdir(database_folder)
            
        if os.path.exists(line_list_folder) == False:
            os.mkdir(line_list_folder)
        
    elif (database in ['HITRAN', 'HITEMP']):
    
        molecule_folder = input_folder + '/' + moleculeName(mol_ID) + '  ~  '
        
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
        
        molecule_folder += iso_name
        
        line_list_folder = molecule_folder + '/' + database + '/'
        
        if os.path.exists(molecule_folder) == False:
            os.mkdir(molecule_folder)
            
        # If we don't remove an existing HITRAN folder, we encounter a Lonely Header exception from hapi.py
        if (database == 'HITRAN'):
            if os.path.exists(line_list_folder):   
                shutil.rmtree(line_list_folder)
            os.mkdir(line_list_folder)
            
        else:
            if os.path.exists(line_list_folder) == False:
                os.mkdir(line_list_folder)
        
    elif (database == 'VALD'):
        
        roman_num = ''
        for i in range(ionization_state):
            roman_num += 'I'
            
        fname = molecule + '_' + roman_num + '.h5'
        
        molecule_folder = input_folder + '/' + molecule + '  ~  (' + roman_num + ')'
        line_list_folder = molecule_folder + '/' + database + '/'
    
        if os.path.exists(molecule_folder) == False:
            os.mkdir(molecule_folder)
            
        if os.path.exists(line_list_folder) == False:
                os.mkdir(line_list_folder)
            
        # Copy the VALD line list file to the newly created folder
        shutil.copy(VALD_data_dir + fname, line_list_folder + '/') 

    
    return line_list_folder
     

def calc_num_ExoMol_trans(html_tags):
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


def create_ExoMol_tag_array(url, broad_URL):
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
    broad_tags = soup2.find_all('a', href = re.compile("[.]broad"))
    pf_tags = soup.find_all('a', href = re.compile("[.]pf"))
    states_tags = soup.find_all('a', href = re.compile("[.]states"))
    trans_tags = soup.find_all('a', href = re.compile("[.]trans"))

    combined_tags = broad_tags + pf_tags + states_tags + trans_tags
    
    return combined_tags

def iterate_ExoMol_tags(tags, host, l_folder, line_list):
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
    num_trans = calc_num_ExoMol_trans(tags)
    
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
        download_ExoMol_file(url, filename, l_folder)
   
        
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
            isotope = ExoMol.get_default_iso(molecule)
        if database in ['hitran', 'hitemp']:
            molecule_dict = HITRAN.create_id_dict()
            mol_id = molecule_dict.get(molecule)
            isotope = isotopologueName(mol_id, 1)
            isotope = HITRAN.replace_iso_name(isotope)
    
    if database == 'vald':
        ion_roman = ''
        for i in range(ionization_state):
            ion_roman += 'I'
    
    if linelist == 'default':
        if database == 'exomol':
            temp_isotope = re.sub('[(]|[)]', '', isotope)
            linelist = ExoMol.get_default_linelist(molecule, temp_isotope)
        if database == 'hitran':
            linelist = 'HITRAN'
        if database == 'hitemp':
            linelist = 'HITEMP'
        if database == 'vald':
            linelist = 'VALD'
            
    if database == 'vald':
        tag = ion_roman
    else:
        tag = isotope
            
    if (database == 'exomol'):
        input_directory = (input_dir + molecule + '  ~  (' + tag + ')/' +
                           'ExoMol' + '/' + linelist + '/')
    else:
        input_directory = (input_dir + molecule + '  ~  (' + tag + ')/' + 
                           linelist + '/')

    if os.path.exists(input_directory):
        return input_directory
    
    else:
        print("You don't seem to have a local folder with the parameters you entered.\n") 
        
        if not os.path.exists(input_dir + '/'):
            print("----- You entered an invalid input directory into the cross_section() function. Please try again. -----")
            sys.exit(0)
        
        elif not os.path.exists(input_dir + '/' + molecule + '  ~  (' + tag + ')/'):
            print("----- There was an error with the molecule + isotope you entered. Here are the available options: -----\n")
            for folder in os.listdir(input_dir + '/'):
                if not folder.startswith('.'):
                    print(folder)
            sys.exit(0)
        
        else:
            print("There was an error with the line list. These are the linelists available: \n")
            for folder in os.listdir(input_dir + '/' + molecule + '  ~  (' + tag + ')/'):
                if not folder.startswith('.'):
                    print(folder)
            sys.exit(0)


def parse_directory(directory, database):
    """
    Determine which linelist and isotopologue this directory contains data for (assumes data was downloaded using our script)

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
    linelist = os.path.basename(directory_name)
    
    # For ExoMol, have to go up one more directory to reach molecule folder
    if (database == 'exomol'):
        directory_name = os.path.dirname(directory_name)
        directory_name = os.path.dirname(directory_name)   
    else:
        directory_name = os.path.dirname(directory_name)

    molecule = os.path.basename(directory_name)
 #   same_molecule = copy.deepcopy(molecule)  # Make a copy of the string because we'll need it for the isotopologue
 #   molecule = re.sub('[  ~].+', '', molecule)  # Keep molecule part of the folder name
    isotopologue = re.sub('.+[  ~]', '', molecule) # Keep isotope part of the folder name
    
    return linelist, isotopologue
    
