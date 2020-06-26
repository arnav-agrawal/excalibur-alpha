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
import Process_linelist
from hapi import partitionSum

def determine_parameters_ExoMol():
    """
    Determines the desired molecular line list from ExoMol from user input. Returns user-inputted values in a tuple.

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
            molecule = input('What molecule would you like to download the line list for?\n')
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
        except ValueError:
            print("\n ----- Please enter an integer for the molecule ID number -----")
        else:
            break
    
    while True:
        try:
            isotopologue_ID = int(input("What is the isotopologue ID of the isotopologue you would like to download?\n"))
        except ValueError:
            print("\n ----- Please enter an integer for the isotopologue ID number -----")
        else:
            break
        
    try:
        partitionSum(molecule_ID, isotopologue_ID, [70, 80], step = 1.0)
    except KeyError:
        print("\n ----- This molecule/isotopologue ID combination is not valid. Please check to make sure the combo you want exists on HITRAN. ----- ")
        sys.exit(0)
    else:
        return molecule_ID, isotopologue_ID


def determine_parameters_VALD():
    return
            
            
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
        print("\n ----- These are not valid ExoMol parameters. Please try again. -----")
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
        print("\n ----- This molecule/isotopologue ID combination is not valid in HITRAN. Please try again. ----- ")
        sys.exit(0)

def type_parameters_VALD():
    return
    

def summon(user_friendly = True, data_base = '', molecule = '', isotope = 'default', linelist = 'default'):
    
    if user_friendly:
        
        while True:
            database = input('What database are you downloading a line list from (ExoMol, HITRAN, or VALD)?\n')
            database = database.lower()
            if database != 'exomol' and database != 'hitran' and database != 'vald' and database != 'hitemp':
                print("\n ----- This is not a supported database, please try again ----- ")
                continue
            else:
                break
        
        if database == 'exomol': 
            mol, iso, lin, URL = determine_parameters_ExoMol()
            line_list_folder = Download_ExoMol.summon_ExoMol(mol, iso, lin, URL)
            Process_linelist.process_file(database, line_list_folder)
            
        if database == 'hitran':
            mol, iso = determine_parameters_HITRAN()
            line_list_folder, abundance = Download_HITRAN.summon_HITRAN(mol, iso)
            Process_linelist.process_file(database, line_list_folder, abundance)
            
        if database == 'vald':
            return
        
        if database == 'hitemp':
            return
        
    if not user_friendly:
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
            line_list_folder = Download_ExoMol.summon_ExoMol(mol, iso, lin, URL)
            
        elif db == 'hitran':
            if isotope == 'default':
                iso = 1
            check_HITRAN(molecule, isotope)
            line_list_folder, abundance = Download_HITRAN.summon_HITRAN(mol, iso)
            Process_linelist.process_file(database, line_list_folder, abundance)
            
        elif db == 'vald':
            return
        
        elif db == 'hitemp':
            return
        
        else:
            print("\n ----- You have not passed in a valid database. Please try again. ----- ")
            sys.exit(0)
        
    
    """
    1. will probably ask the user if they want to do the "advanced version" of download or the "easy version"
    2. Easy Version... ask the user for the molecule, isotopologue, etc.
    3. Hard version... user can manually set the parameters in
    4. use one of the three downloader functions based on the file type that the user specifies
    5. Process the downloaded files
    6. Run Cthulhu
    
    

    Returns
    -------
    None.

    """