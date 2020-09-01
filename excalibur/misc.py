import numpy as np
import pandas as pd
import re


def check_molecule(molecule):
    """
    Check if the given string is a molecule

    Parameters
    ----------
    molecule : String
        Molecular formula.

    Returns
    -------
    True if the given string is a molecule, false otherwise (if it is an atom).

    """
    match = re.match('^[A-Z]{1}[a-z]?$', molecule)     # Matches a string containing only 1 capital letter followed by 0 or 1 lower case letters
    
    if match: return False   # If our 'molecule' matches the pattern, it is really an atom
    else: return True        # We did not get a match, therefore must have a molecule


def write_output(output_directory, molecule, T, log_P, nu_out, sigma_out):
            
    f = open(output_directory + str(molecule) + '_T' + str(T) + 'K_log_P' + str(log_P) + '_sigma.txt','w')
                    
    for i in range(len(nu_out)):
        f.write('%.8f %.8e \n' %(nu_out[i], sigma_out[i]))
                        
    f.close()
    
    return nu_out, sigma_out

