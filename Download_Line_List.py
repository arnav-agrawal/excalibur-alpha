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
import requests
import Download_ExoMol
import Download_HITRAN
import Download_HITEMP
from hapi import partitionSum, moleculeName

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
            response = requests.get(website + molecule + '/')
            response.raise_for_status() #Raises HTTPError if a bad request is made (server or client error)
            
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
    
    while True:
        try:
            molecule_ID = int(input("What molecule would you like to download the line list for? Please enter the molecule ID number (left-hand column) found here: https://hitran.org/lbl/# \n"))
            partitionSum(molecule_ID, 1, [70, 80], step = 1.0)
        except ValueError:
            print("\n ----- Please enter an integer for the molecule ID number -----")
        except KeyError:
            print("\n ----- This molecule ID is not valid. Please check to make sure the molecule you want exists on HITRAN. ----- ")
        else:
            break
    
    while True:
        try:
            isotopologue_ID = int(input("What is the isotopologue ID of the isotopologue you would like to download?\n"))
            partitionSum(molecule_ID, isotopologue_ID, [70, 80], step = 1.0)
        except ValueError:
            print("\n ----- Please enter an integer for the isotopologue ID number -----")
        except KeyError:
            print("\n ----- This molecule/isotopologue ID combination is not valid. Please check to make sure the combo you want exists on HITRAN. ----- ")
        else:
            return molecule_ID, isotopologue_ID


def determine_parameters_VALD():
    return


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
    
    table = Download_HITEMP.HITEMP_table()
    
    while True:
        try:
            molecule_ID = int(input("What molecule would you like to download the line list for? Please enter the molecule ID number (left-hand column) found here: https://hitran.org/lbl/# \n"))
            if molecule_ID in table['ID'].values:
                row = table.loc[table['ID'] == molecule_ID]   # Find the DataFrame row that contains the molecule_ID
                break
        except ValueError:
            print("\n ----- Please enter an integer for the molecule ID number -----")
        else:
            print("\n ----- This molecule ID is not valid. Please check to make sure the molecule you want exists on HITEMP. -----")
    
    while True:
        try:
            isotopologue_ID = int(input("What is the isotopologue ID of the isotopologue you would like to download?\n"))
            isotope_count = row.loc[row.index.values[0], 'Iso Count'] # Find the number of isotopologues of the given molecule ID
            if isotopologue_ID <= isotope_count:
                break
        except ValueError:
            print("\n ----- Please enter an integer for the isotopologue ID number -----")
        else:
            print("\n ----- This molecule/isotopologue ID combination is not valid. Please check to make sure the combo you want exists on HITEMP. ----- ")
        
    return molecule_ID, isotopologue_ID
            
            
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
        requests.get(website)
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
    
    try:
        partitionSum(molecule, isotope, [70, 80], step = 1.0)
    except KeyError:
        print("\n ----- This molecule/isotopologue ID combination is not valid in HITRAN. Please try calling the summon() function again. Make sure you enter the ID number of the molecule you want. ----- ")
        print("\n ----- A list of supported molecule IDs can be found here: https://hitran.org/lbl/ -----")
        sys.exit(0)


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
    
    table = Download_HITEMP.HITEMP_table()
    if molecule in table['ID'].values:
        row = table.loc[table['ID'] == molecule]
        isotope_count = row.loc[row.index.values[0], 'Iso Count']
        if isotope <= isotope_count:
            return
    else:
        print("\n ----- This molecule/isotopologue ID combination is not valid in HITEMP. Please try calling the summon() function again. Make sure you enter the ID number of the molecule you want. ----- ")
        print("\n ----- A list of supported molecule IDs can be found here: https://hitran.org/hitemp/ -----")
        sys.exit(0)

def check_VALD():
    return
    

def summon(user_friendly = True, data_base = '', molecule = '', isotope = 'default', linelist = 'default'):
    """
    Makes callls to other downloader files to retrieve the data from the desired database

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
            return
        
        if database == 'hitemp':
            mol, iso = determine_parameters_HITEMP()
            Download_HITEMP.summon_HITEMP(mol, iso)
            
        
    if not user_friendly: # If the user just wants to call the function with parameters directly passed in
        db = data_base.lower()
        mol = molecule
        iso = isotope
        lin = linelist
            
        if db == 'exomol':
            if isotope == 'default':
                check_ExoMol(mol)
                iso = Download_ExoMol.get_default_iso(mol)
            if linelist == 'default':
                check_ExoMol(mol, iso)
                lin = Download_ExoMol.get_default_linelist(mol, iso)
            check_ExoMol(molecule, isotope, linelist)
            URL = "http://exomol.com/data/molecules/" + mol + '/' + iso + '/' + lin + '/'
            Download_ExoMol.summon_ExoMol(mol, iso, lin, URL)
            
        elif db == 'hitran':
            if isotope == 'default':
                iso = 1
            check_HITRAN(molecule, isotope)
            Download_HITRAN.summon_HITRAN(mol, iso)
            
        elif db == 'vald':
            return
        
        elif db == 'hitemp':
            if isotope == 'default':
                iso = 1
            check_HITEMP(molecule, isotope)
            Download_HITEMP.summon_HITEMP(mol, iso)
        
        else:
            print("\n ----- You have not passed in a valid database. Please try calling the summon() function again. ----- ")
            sys.exit(0)
        
    
    print("\nDownload complete.")