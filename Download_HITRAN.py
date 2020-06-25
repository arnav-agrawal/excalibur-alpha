#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:07:01 2020

@author: arnav
"""

#Can use ISO_ID later if needed which gives info about all the molecule/isotope combinations in HITRAN

from hapi import *
import sys
import os


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
    molecule_folder = input_folder + '/' + moleculeName(mol_ID) + '-' + isotopologueName(mol_ID, iso_ID)
    line_list_folder = molecule_folder + '/HITRAN'
    
    if os.path.exists(input_folder) == False:
        os.mkdir(input_folder)
    
    if os.path.exists(molecule_folder) == False:
        os.mkdir(molecule_folder)

    if os.path.exists(line_list_folder) == False:
        os.mkdir(line_list_folder)
        
    return line_list_folder


def create_pf(mol_ID, iso_ID, folder, T_min = 70, T_max = 3001, step = 1.0):
    T, Q = partitionSum(mol_ID, iso_ID, [T_min, T_max], step)

    out_file = folder + '/' + moleculeName(mol_ID) + '.pf'
    f_out = open(out_file, 'w')
    f_out.write('T | Q \n') 
    for i in range(len(T)):
        f_out.write('%.1f %.4f \n' %(T[i], Q[i]))
        

def download_trans_file(mol_ID, iso_ID, folder, nu_min = 200, nu_max = 25000):
    db_begin(folder)
    fetch(moleculeName(mol_ID), mol_ID, iso_ID, nu_min, nu_max)
    

def summon_HITRAN(molecule, isotopologue):
    output_folder = create_directories(molecule, isotopologue)
    create_pf(molecule, isotopologue, output_folder)
    download_trans_file(molecule, isotopologue, output_folder)
    
    molecular_abundance = abundance(molecule, isotopologue)  #Used for processing the .data file later
    
    return output_folder, molecular_abundance