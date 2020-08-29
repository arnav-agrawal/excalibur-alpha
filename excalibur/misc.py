import numpy as np
import pandas as pd

def write_output(output_directory, molecule, T, log_P, nu_out, sigma_out):
            
    f = open(output_directory + str(molecule) + '_T' + str(T) + 'K_log_P' + str(log_P) + '_sigma.txt','w')
                    
    for i in range(len(nu_out)):
        f.write('%.8f %.8e \n' %(nu_out[i], sigma_out[i]))
                        
    f.close()
    
    return nu_out, sigma_out