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
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter


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
    
    print("\nPlotting the cross-section of", molecule, "at", temperature, "K and", np.power(10.0, log_pressure), "bar")
    
    # Make wavenumber plot
    plt.figure()
    plt.clf()
    ax = plt.gca()
    
    plt.semilogy(nu_out, sigma_out, lw=0.3, alpha = 0.5, color = 'red', label = (molecule + r'Cross Section (out)'))   
    
    plt.xlim([200.0, 25000.0])
    plt.ylim([1.0e-30, 1.0e-12])

    plt.ylabel(r'Cross Section (cm$^2$)')
    plt.xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)')

    legend = plt.legend(loc='upper left', shadow=False, frameon=False, prop={'size':6})
    
    plt.savefig('../plots/' + molecule + '_' + str(temperature) + 'K_' + str(np.power(10.0, log_pressure)) + 'bar_nu.pdf')
    
    plt.close()
    
    # Make wavelength plot
    plt.clf()
    ax = plt.gca()
    
    xmajorLocator   = MultipleLocator(0.2)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.02)
    
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    
    plt.loglog(wl_out, sigma_out, lw=0.3, alpha = 0.5, color= 'red', label = (molecule + r'$\mathrm{\, \, Cross \, \, Section}$')) 
    
    plt.ylim([1.0e-30, 1.0e-14])
    plt.xlim([0.4, 10.0])
    
    plt.ylabel(r'Cross Section (cm$^2$)')
    plt.xlabel(r'Wavelength (μm)')
    
    ax.text(0.7, 5.0e-16, (r'T = ' + str(temperature) + r'K, P = ' + str(np.power(10.0, log_pressure)) + r'bar'), fontsize = 10)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':7})
    
    #plt.show()

    plt.savefig('../plots/' + molecule + '_' + str(temperature) + 'K_' + str(np.power(10.0, log_pressure)) + 'bar.pdf')
    
    print("\nPlotting complete.")
