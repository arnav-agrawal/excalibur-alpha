import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, \
                              FuncFormatter, ScalarFormatter, NullFormatter


def plot_sigma_wl(species, temperature, log_pressure, nu_arr = [], sigma_arr = [], 
                  file = '', database = '', plot_dir = './plots/', **kwargs):
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
  #  if nu_arr == [] and sigma_arr == [] and file == '':
  #      print("----- You have not specified enough arguments for this function. ----- ")
  #      sys.exit(0)
     
    # User specified a file    
  #  if file != '':
  #      try:
  #          data = pd.read_fwf(file, widths = [13, 14], names = ['Nu Out', 'Sigma Out'], header = None)
  #          nu = data['Nu Out']
  #          sigma = data['Sigma Out']
    
  #          wl = 1.0e4/nu
            
  #      except TypeError:
  #          print("----- You did not pass in a valid file. ----- ")
  #          sys.exit(0)
    
    # User passed in wl and sigma arr        
 #   if nu_arr != [] and sigma_arr != []:
 #       nu = nu_arr
 #       sigma = sigma_arr
 #       wl = 1.0e4/nu
        
    if not os.path.exists('./plots/'):
        os.mkdir('./plots')
       
    
    nu_plt = nu_arr
    wl_plt = 1.0e4/nu_plt
    sigma_plt = sigma_arr + 1.0e-250   # Add small value to avoid log(0) on log plots
    
    
    pressure = np.power(10.0, log_pressure)
    
    print("Plotting the cross-section of", species, "at", temperature, "K and", pressure, "bar")
        
    #***** Make wavenumber plot *****#
  #  fig = plt.figure()
  #  ax = plt.gca()
    
  #  ax.set_yscale("log")
    
  #  ax.plot(nu_plt, sigma_plt, lw=0.3, alpha = 0.5, color = 'crimson', label = (molecule + r' Cross Section'))   
    
  #  ax.set_ylim([1.0e-30, np.log10(np.max(sigma_plt)) + 2.0])
  #  ax.set_xlim([200.0, 25000.0])

  #  ax.set_ylabel(r'Cross Section (cm$^2$)', size = 14)
  #  ax.set_xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)', size = 14)

  #  legend = plt.legend(loc='upper left', shadow=False, frameon=False, prop={'size':10})
    
  #  plt.tight_layout()
    
  #  plt.savefig('./plots/' + molecule + '_' + str(temperature) + 'K_' + str(pressure) + 'bar_nu.pdf')
    
  #  plt.close()
    
    
    #***** Make wavelength plot *****#
    fig = plt.figure()
    ax = plt.gca()
    
    min_sigma = 1.0e-30
    max_sigma = 10.0**(np.max(np.ceil(np.log10(sigma_plt) / 2.0) * 2.0))
    
    ax.loglog(wl_plt, sigma_plt, lw=0.3, alpha = 0.5, color= 'crimson', label = (species + r' Cross Section')) 
    
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    ax.set_xticklabels(['0.4', '0.6', '0.8', '1', '2', '4', '6', '8', '10'])
    
    ax.set_ylim([min_sigma, max_sigma])
 #   ax.set_ylim([1.0e-30, round(np.log10(np.max(sigma)) + 2.0)])
    ax.set_xlim([0.4, 10.0])
    
    ax.set_ylabel(r'Cross Section (cm$^2$)', size = 14)
    ax.set_xlabel(r'Wavelength (μm)', size = 14)
    
    ax.text(0.45, 10**(np.log10(max_sigma) - 1.0), (r'T = ' + str(temperature) + r' K, P = ' + str(pressure) + r' bar'), fontsize = 10)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':10})
    
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)
        
    plt.tight_layout()

    plt.savefig(plot_dir + species + '_' + str(temperature) + 'K_' +
                str(pressure) + 'bar_' + database + '.pdf')
    
    print("\nPlotting complete.")


def compare_cross_sections(molecule, label_1, label_2, nu_arr_1 = [], nu_arr_2 = [], 
                           sigma_arr_1 = [], sigma_arr_2 = [], **kwargs):

    wl_1 = 1.0e4/nu_arr_1
    wl_2 = 1.0e4/nu_arr_2
        
    if not os.path.exists('./plots/'):
        os.mkdir('./plots')
    
    print("\nComparing cross-sections of", molecule)
    
    #***** Make wavenumber plot *****#
    fig = plt.figure()
    ax = plt.gca()
    
    ax.set_yscale("log")
 #   ax.set_xscale("log")
 
    ax.plot(nu_arr_1, sigma_arr_1, lw=0.3, alpha = 0.5, color= 'crimson', label = (molecule + r' Cross Section ' + label_1)) 
    ax.plot(nu_arr_2, sigma_arr_2, lw=0.3, alpha = 0.5, color= 'royalblue', label = (molecule + r' Cross Section ' + label_2)) 
    
 #   ax.set_xlim([1800.2, 1804.1])
 #   ax.set_ylim([3.0e-22, 3.0e-18])
    ax.set_xlim([995.0, 1005.0])  
    ax.set_ylim([1.0e-23, 1.0e-20])

    
    ax.set_ylabel(r'Cross Section (cm$^2$)', size = 14)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', size = 14)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':10})
    
    plt.tight_layout()

    plt.savefig('./plots/' + molecule + '_compare_' + label_1 + '_' + label_2 + '_Region_2.pdf')
    
    if (1 == 2):
    
        #***** Make wavelength plot *****#
        fig = plt.figure()
        ax = plt.gca()
        
        ax.set_yscale("log")
        ax.set_xscale("log")
        
        min_sigma = 1.0e-30
        max_sigma = 10.0**(np.max(np.ceil(np.log10(sigma_arr_1 + 1.0e-250) / 2.0) * 2.0))
        
        ax.plot(wl_1, sigma_arr_1, lw=0.3, alpha = 0.5, color= 'crimson', label = (molecule + r' Cross Section ' + label_1)) 
        ax.plot(wl_2, sigma_arr_2, lw=0.3, alpha = 0.5, color= 'royalblue', label = (molecule + r' Cross Section ' + label_2)) 
        
        ax.set_xticks([0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        ax.set_xticklabels(['0.4', '0.6', '0.8', '1', '2', '4', '6', '8', '10'])
        
     #   ax.set_ylim([min_sigma, max_sigma])
     #   ax.set_xlim([0.4, 10])
     #   ax.set_ylim([1.0e-25, 1.0e-18])
     #   ax.set_xlim([1.1, 1.7])
     
        ax.set_ylim([8.0e-25, 1.0e-22])
        ax.set_xlim([1.2, 1.201])
    
        
        ax.set_ylabel(r'Cross Section (cm$^2$)', size = 14)
        ax.set_xlabel(r'Wavelength (μm)', size = 14)
        
    #    ax.text(0.5, 5.0e-12, (r'T = ' + str(temperature) + r' K, P = ' + str(pressure) + r' bar'), fontsize = 10)
        
        legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':10})
        
        plt.tight_layout()
    
        plt.savefig('./plots/' + molecule + '_comparison_' + label_1 + '_' + label_2 + '_zoom_a0.pdf')
    
    print("\nPlotting complete.")