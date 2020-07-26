#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:29:06 2020

@author: arnav
"""

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, \
                              FuncFormatter, ScalarFormatter, NullFormatter


def plot_results(molecule, temperature, log_pressure, nu_arr = [], sigma_arr = [], file = '', **kwargs):
    """
    Generate a plot of a cross_section file, in both wavelength and wavenumber

    Parameters
    ----------
    molecule : TYPE
        DESCRIPTION.
    temperature : TYPE
        DESCRIPTION.
    log_pressure : TYPE
        DESCRIPTION.
    nu_arr : TYPE, optional
        DESCRIPTION. The default is [].
    sigma_arr : TYPE, optional
        DESCRIPTION. The default is [].
    file : TYPE, optional
        DESCRIPTION. The default is ''.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # User did not specify a wl and sigma array and also did not specify a file
    if nu_arr == [] and sigma_arr == [] and file == '':
        print("----- You have not specified enough arguments for this function. ----- ")
        sys.exit(0)
     
    # User specified a file    
    if file != '':
        try:
            data = pd.read_fwf(file, widths = [13, 14], names = ['Nu Out', 'Sigma Out'], header = None)
            nu_out = data['Nu Out']
            sigma_out = data['Sigma Out']
    
            wl_out = 1.0e4/nu_out
            
        except TypeError:
            print("----- You did not pass in a valid file. ----- ")
            sys.exit(0)
    
    # User passed in wl and sigma arr        
    if nu_arr != [] and sigma_arr != []:
        nu_out = nu_arr
        sigma_out = sigma_arr
        wl_out = 1.0e4/nu_out
        
    if not os.path.exists('../plots/'):
        os.mkdir('../plots')
    
    pressure = np.power(10.0, log_pressure)
    
    print("\nPlotting the cross-section of", molecule, "at", temperature, "K and", pressure, "bar")
        
    #***** Make wavenumber plot *****#
    fig = plt.figure()
    ax = plt.gca()
    
    ax.set_yscale("log")
    
    ax.plot(nu_out, sigma_out, lw=0.3, alpha = 0.5, color = 'crimson', label = (molecule + r' Cross Section'))   
    
    ax.set_ylim([1.0e-30, 1.0e-14])
    ax.set_xlim([200.0, 25000.0])

    ax.set_ylabel(r'Cross Section (cm$^2$)', size = 14)
    ax.set_xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)', size = 14)

    legend = plt.legend(loc='upper left', shadow=False, frameon=False, prop={'size':10})
    
    plt.tight_layout()
    

    plt.savefig('../plots/' + molecule + '_' + str(temperature) + 'K_' + str(pressure) + 'bar_nu.pdf')
    
    plt.close()
    
    #***** Make wavelength plot *****#
    fig = plt.figure()
    ax = plt.gca()
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    
    ax.plot(wl_out, sigma_out, lw=0.3, alpha = 0.5, color= 'crimson', label = (molecule + r' Cross Section')) 
    
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    ax.set_xticklabels(['0.4', '0.6', '0.8', '1', '2', '4', '6', '8', '10'])
    
    ax.set_ylim([1.0e-30, 1.0e-14])
    ax.set_xlim([0.4, 10.0])
    
    ax.set_ylabel(r'Cross Section (cm$^2$)', size = 14)
    ax.set_xlabel(r'Wavelength (μm)', size = 14)
    
    ax.text(0.5, 5.0e-16, (r'T = ' + str(temperature) + r' K, P = ' + str(pressure) + r' bar'), fontsize = 10)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':10})
    
    plt.tight_layout()

    plt.savefig('../plots/' + molecule + '_' + str(temperature) + 'K_' + str(pressure) + 'bar.pdf')
    
    print("\nPlotting complete.")
