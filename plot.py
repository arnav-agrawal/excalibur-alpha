#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:29:06 2020

@author: arnav
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter


def plot_results(file, **kwargs):
    
    data = pd.read_fwf(file, widths = [13, 14], names = ['Nu Out', 'Sigma Out'], header = None)

            
    nu_out = data['Nu Out']
    sigma_out = data['Sigma Out']
    species = 'HCN'
    T = 1000
    P = 0
    
    wl_out = 1.0e4/nu_out
    
    # Make wavenumber plot
    plt.figure()
    plt.clf()
    ax = plt.gca()
    
    plt.semilogy(nu_out, sigma_out, lw=0.3, alpha = 0.5, color = 'red', label = (species + r'Cross Section (out)'))   
    
    plt.xlim([200.0, 25000.0])
    plt.ylim([1.0e-30, 1.0e-12])

    plt.ylabel(r'Cross Section (cm$^2$)')
    plt.xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)')

    legend = plt.legend(loc='upper left', shadow=False, frameon=False, prop={'size':6})
    
    plt.savefig('../output/' + species + '_' + str(T) + 'K_' + str(P) + 'bar_nu.pdf')
    
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
    
    plt.loglog(wl_out, sigma_out, lw=0.3, alpha = 0.5, color= 'red', label = (species + r'$\mathrm{\, \, Cross \, \, Section}$')) 
    
    plt.ylim([1.0e-30, 1.0e-14])
    plt.xlim([0.4, 10.0])
    
    plt.ylabel(r'Cross Section (cm$^2$)')
    plt.xlabel(r'Wavelength (μm)')
    
    ax.text(0.7, 5.0e-16, (r'T = ' + str(T) + r'K, P = ' + str(P) + r'bar'), fontsize = 10)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':7})
    
    #plt.show()

    plt.savefig('../output/' + species + '_' + str(T) + 'K_' + str(P) + 'bar.pdf')
