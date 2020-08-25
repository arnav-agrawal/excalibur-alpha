#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:33:33 2020

@author: arnav
"""

"""
Import download_Exomol, download_HITRAN, and download_VALD
"""

import sys
import os
import requests
import re
from . import Download_ExoMol
from . import Download_HITRAN
from . import Download_HITEMP
from . import Download_VALD
from .hapi import partitionSum, moleculeName

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

def determine_parameters_ExoMol():
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
                isotopologue = Download_ExoMol.get_default_iso(molecule)
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
                linelist = Download_ExoMol.get_default_linelist(molecule, isotopologue)
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
        

def determine_parameters_HITRAN():
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


def determine_parameters_VALD():
    
    while True:
        molecule = input("What atom would you like to download the line list for? \n")
        fname = molecule + '_I.h5'  # Check if at least the neutral version of this atom is supported (i.e. that we even provide the line list for this atom)
        if fname in os.listdir('../VALD Line Lists'): 
            break
        else:
            print("\n ----- The VALD line list for this atom does not exist. Please try again. -----")
    
    while True:
        try:
            ionization_state = int(input("What ionization state of this atom would you like to download the line list for? ('1' is the neutral state) \n"))
        
        except ValueError:
             print("\n ----- Please enter an integer for the ionization state -----")
        
        else:
            roman_num = ''
            for i in range(ionization_state):
                roman_num += 'I'
        
            fname = molecule + '_' + roman_num + '.h5'  # Check if at least the neutral version of this atom is supported (i.e. that we even provide the line list for this atom)
            if fname in os.listdir('../VALD Line Lists'): 
                return molecule, ionization_state
            else:
                print("\n ----- The VALD line list for this atom/ionization state combination does not exist. Please try again. -----")


def determine_parameters_HITEMP():
    """
    Determines the molecule and isotopologue that would like to be accessed based on user input

    Returns
    -------
    molecule_ID : int
        DESCRIPTION.
    isotopologue_ID : int
        DESCRIPTION.

    """
    
    molecule_dict = create_id_dict()
    table = Download_HITEMP.HITEMP_table()
    
    while True:
        molecule = input("What molecule would you like to download the line list for? \n")
        molecule_id_number = molecule_dict.get(molecule)
        if molecule_id_number in table['ID'].values:
            row = table.loc[table['ID'] == molecule_id_number]   # Find the DataFrame row that contains the molecule_ID
            break   
        else:
            print("\n ----- This molecule ID is not valid. Please check to make sure the molecule you want exists on HITEMP. -----")
    
    while True:
        try:
            isotopologue_ID = int(input("What is the isotopologue ID of the isotopologue you would like to download? Enter '1' for the most abundant isotopologue, '2' for next most abundant, etc. More info found here: https://hitran.org/lbl/# \n"))
            isotope_count = row.loc[row.index.values[0], 'Iso Count'] # Find the number of isotopologues of the given molecule ID
            if isotopologue_ID <= isotope_count:
                break
        except ValueError:
            print("\n ----- Please enter an integer for the isotopologue ID number -----")
        else:
            print("\n ----- This molecule/isotopologue ID combination is not valid. Please check to make sure the combo you want exists on HITEMP. ----- ")
        
    return molecule_id_number, isotopologue_ID
            
            
def check_ExoMol(molecule, isotope = '', linelist = ''):
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
        

def check_HITRAN(molecule, isotope):
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


def check_HITEMP(molecule, isotope):
    """
    Checks if the parameters passed into the summon() function by the user are valid to use with the HITEMP database

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
    
    if molecule_id_number is None:
        print("\n ----- This molecule/isotopologue ID combination is not valid in HITEMP. Please try calling the summon() function again. Make sure you enter the ID number of the molecule you want. ----- ")
        print("\n ----- A list of supported molecule IDs can be found here: https://hitran.org/hitemp/ -----")
        sys.exit(0)
    
    table = Download_HITEMP.HITEMP_table()
    if molecule_id_number in table['ID'].values:
        row = table.loc[table['ID'] == molecule_id_number]
        isotope_count = row.loc[row.index.values[0], 'Iso Count']
        if isotope <= isotope_count:
            return molecule_id_number
    else:
        print("\n ----- This molecule/isotopologue ID combination is not valid in HITEMP. Please try calling the summon() function again. Make sure you enter the ID number of the molecule you want. ----- ")
        print("\n ----- A list of supported molecule IDs can be found here: https://hitran.org/hitemp/ -----")
        sys.exit(0)
        

def check_VALD(mol, ion):
    roman_num = ''
    for i in range(ion):
            roman_num += 'I'
    fname = mol + '_' + roman_num + '.h5'
    if fname not in os.listdir('../VALD Line Lists'): 
        print("\n ----- The VALD line list for this atom/isotope combination does not exist. Please try again. -----")
        sys.exit(0)
        
    

def summon(database = '', molecule = '', isotope = 'default', linelist = 'default', ionization_state = 1, **kwargs):
    """
    Makes calls to other downloader files to retrieve the data from the desired database

    Parameters
    ----------
    user_friendly : TYPE, optional
        DESCRIPTION. The default is True.
    data_base : TYPE, optional
        DESCRIPTION. The default is ''.
    molecule : TYPE, optional
        DESCRIPTION. The default is ''.
    isotope : TYPE, optional
        DESCRIPTION. The default is 'default'.
    linelist : TYPE, optional
        DESCRIPTION. The default is 'default'.

    Returns
    -------
    None.

    """
    
    if database != '' and molecule != '': user_friendly = False
    else: user_friendly = True
        
    
    if user_friendly: # If the user wants to be guided via terminal prompts
        
        while True:
            database = input('What database are you downloading a line list from (ExoMol, HITRAN, HITEMP, or VALD)?\n')
            database = database.lower()
            if database == 'exomol' or database == 'hitran' or database == 'vald' or database == 'hitemp':
                break
            else:
                print("\n ----- This is not a supported database, please try again ----- ")
                
        
        if database == 'exomol': 
            mol, iso, lin, URL = determine_parameters_ExoMol()
            Download_ExoMol.summon_ExoMol(mol, iso, lin, URL)
            
        if database == 'hitran':
            mol, iso = determine_parameters_HITRAN()
            Download_HITRAN.summon_HITRAN(mol, iso)
            
        if database == 'vald':
            mol, ion = determine_parameters_VALD()
            Download_VALD.summon_VALD(mol, ion)
        
        if database == 'hitemp':
            mol, iso = determine_parameters_HITEMP()
            Download_HITEMP.summon_HITEMP(mol, iso)
            
        
    if not user_friendly: # If the user just wants to call the function with parameters directly passed in
        db = database.lower()
        mol = molecule
        if isinstance(isotope, str):
            try:
                isotope = int(isotope)
            except ValueError:
                pass
        iso = isotope
        lin = linelist
        ion = ionization_state
        
        if db == 'exomol':
            mol = re.sub('[+]', '_p', mol)  # Handle ions
            iso = re.sub('[+]', '_p', iso)  # Handle ions
            
        if db == 'exomol':
            if isotope == 'default':
                check_ExoMol(mol)
                iso = Download_ExoMol.get_default_iso(mol)
            if linelist == 'default':
                check_ExoMol(mol, iso)
                lin = Download_ExoMol.get_default_linelist(mol, iso)

            check_ExoMol(mol, iso, lin)
            URL = "http://exomol.com/data/molecules/" + mol + '/' + iso + '/' + lin + '/'
            Download_ExoMol.summon_ExoMol(mol, iso, lin, URL)
            
        elif db == 'hitran':
            if isotope == 'default':
                iso = 1
            mol = check_HITRAN(molecule, iso)
            Download_HITRAN.summon_HITRAN(mol, iso)
            
        elif db == 'vald':
            check_VALD(mol, ion)
            Download_VALD.summon_VALD(mol, ion)
        
        elif db == 'hitemp':
            if isotope == 'default':
                iso = 1
            mol = check_HITEMP(molecule, iso)
            Download_HITEMP.summon_HITEMP(mol, iso)
        
        else:
            print("\n ----- You have not passed in a valid database. Please try calling the summon() function again. ----- ")
            sys.exit(0)
        
    print("\nLine list ready.\n")
