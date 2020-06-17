import numpy as np
import pandas as pd
import os


#chem_species = 'H2O'
#species_id = '1H2-16O'
linelist_type = 'EXOMOL'
#linelist = 'BT2'
#prefix = './' + species_id + '__'
#prefix2 = './' + linelist +'/'

def process_Exomol_file(prefix_in, prefix_out, file_type, broad_species, linelist, trans_1, trans_2):
    
    if (file_type == '.broad'):
        
        gamma_L_0 = []
        n_L = []
        J = []

        f_in = open(prefix_in + broad_species + '.broad', 'r')

        for line in f_in:
            
            line = line.strip()
            line = line.split()
    
            if (len(line) == 4):
                
                gamma_L_0.append(float(line[1]))
                n_L.append(float(line[2]))
                J.append(float(line[3]))
      
        f_in.close()
    
        f_out = open(prefix_out + broad_species + '.broad','w')
    
        f_out.write('J | gamma_L_0 | n_L \n')
    
        for i in range(len(J)):
            f_out.write('%.1f %.4f %.3f \n' %(J[i], gamma_L_0[i], n_L[i]))
        
        f_out.close()
        
    elif (file_type == '.states'):
        
        n = []
        E = []
        g = []
        J = []

        f_in = open(prefix_in + linelist + '.states', 'r')

        for line in f_in:
            
            line = line.strip()
            line = line.split()
                
            n.append(int(line[0]))
            E.append(float(line[1]))
            g.append(float(line[2]))
            J.append(float(line[3]))
      
        f_in.close()
    
        f_out = open(prefix_out + linelist + '.states','w')
    
        f_out.write('i | E (cm^-1) | g | J \n')
    
        for i in range(len(n)):
            f_out.write('%d %.6f %d %.1f \n' %(n[i], E[i], g[i], J[i]))
        
        f_out.close()
        
    elif (file_type == '.pf'):
        
        T_pf = []
        Q = []

        f_in = open(prefix_in + linelist + '.pf', 'r')

        for line in f_in:
            
            line = line.strip()
            line = line.split()
                
            T_pf.append(float(line[0]))
            Q.append(float(line[1]))
      
        f_in.close()
    
        f_out = open(prefix_out + linelist + '.pf','w')
    
        f_out.write('T | Q \n')
    
        for i in range(len(T_pf)):
            f_out.write('%.1f %.4f \n' %(T_pf[i], Q[i]))
        
        f_out.close()
        
    elif (file_type == '.trans'):
        
        f_in = open(prefix_out + linelist + '__' + trans_1 + '-' + trans_2 + '.trans','r')
        f_out = open(prefix_out + linelist + '__' + trans_1 + '-' + trans_2 + '.trans2','w')
        
        count = 0
        
        for line in f_in:
            
            if (count==0): 
                f_out.write('i | f | A (s^-1) \n')
            
            else:
                line = line.strip()
                line = line.split()
                
                n_i = int(line[0])
                n_f = int(line[1])
                A_fi = float(line[2])
                
                f_out.write('%d %d %.4e \n' %(n_i, n_f, A_fi))
                
            count +=1
      
        f_in.close()
        f_out.close()

"""        
def process_HITRAN_file(file_type, species, trans_file, file_number):
    
    # For undoing HITRAN terrestrial intensity scalings
    X_iso = {'H2O':   0.997317, 'CO2':   0.984204, 'H2O':  0.997317, 'O3':     0.992901,
             'N2O':   0.990333, 'CO':    0.986544, 'CH4':  0.988274, 'O2':     0.995262,
             'NO':    0.993974, 'SO2':   0.945678, 'NO2':  0.991616, 'NH3':    0.995872,
             'HNO3':  0.989110, 'OH':    0.997473, 'HF':   0.999844, 'HCl':    0.757587,
             'HBr':   0.506781, 'HI':    0.999844, 'ClO':  0.755908, 'OCS':    0.937395,
             'H2CO':  0.986237, 'HOCl':  0.755790, 'N2':   0.992687, 'HCN':    0.985114,
             'CH3Cl': 0.748937, 'H2O2':  0.994952, 'C2H2': 0.977599, 'C2H6':   0.976990,
             'PH3':   0.999533, 'COF2':  0.986544, 'SF6':  0.950180, 'H2S':    0.949884,
             'HCOOH': 0.983898, 'HO2':   0.995107, 'O':    0.997628, 'ClONO2': 0.749570,
             'NO+':   0.993974, 'HOBr':  0.505579, 'C2H4': 0.977294, 'CH3OH':  0.985930,
             'CH3Br': 0.500995, 'CH3CN': 0.973866, 'CF4':  0.988890, 'C4H2':   0.955998,
             'HC3N':  0.963346, 'H2':    0.999688, 'CS':   0.939624, 'SO3':    0.943400,
             'C2N2':  0.970752, 'COCl2': 0.566392}   
    
    #field_lengths = [2,1,12,10,10,5,5,10,4,8,5,3,2,2,2,2,6,2,2,2,2,3,2,4,3,2,4,3,3,1]
    #field_lengths = [3,12,10,10,5,5,10,4,8,2,2,2,2,2,2,2,1,3,2,2,2,2,2,2,2,2,1,3,12,1,3,1,25,6,1,6]
    
    if (file_type == '.pf'):
        
        T_pf = []
        Q = []

        f_in = open(prefix + linelist + '.pf', 'r')

        for line in f_in:
            
            line = line.strip()
            line = line.split()
                
            T_pf.append(float(line[0]))
            Q.append(float(line[1]))
      
        f_in.close()
    
        f_out = open(prefix2 + linelist + '.pf','w')
    
        f_out.write('T | Q \n')
    
        for i in range(len(T_pf)):
            f_out.write('%.1f %.4f \n' %(T_pf[i], Q[i]))
        
        f_out.close()
        
        return 0.0, 0.0, 0.0
    
    if (file_type == '.h2'):
        
        field_lengths = [2,1,12,10,10,5,5,10,4,8,15,15,15,6,3]
        
        broad_file_name = [filename for filename in os.listdir('.') if filename.endswith(".h2")]
        broad_file = pd.read_fwf(broad_file_name[0], widths=field_lengths, header=None)
        
        gamma_h2 = np.array(broad_file[5])
        n_h2 = np.array(broad_file[8])
        J_lower = np.array(broad_file[14])
        
        J_max = max(J_lower)
        J_sorted = np.arange(int(J_max))
        
        gamma_h2_avg, n_h2_avg = (np.array([]) for _ in range(2))
        
        for i in range(int(J_max)):
            
            gamma_h2_i = np.mean(gamma_h2[np.where(J_lower == J_sorted[i])])
            n_h2_i = np.mean(n_h2[np.where(J_lower == J_sorted[i])])
            
            gamma_h2_avg = np.append(gamma_h2_avg, gamma_h2_i)
            n_h2_avg = np.append(n_h2_avg, n_h2_i)
        
        # Write H2 broadening file
        f_out = open('./H2.broad','w')
            
        f_out.write('J | gamma_L_0 | n_L \n')
            
        for i in range(len(J_sorted)):
            f_out.write('%.1f %.4f %.3f \n' %(J_sorted[i], gamma_h2_avg[i], n_h2_avg[i]))
                
        f_out.close()
        
    if (file_type == '.he'):
        
        field_lengths = [2,1,12,10,10,5,5,10,4,8,15,15,15,6,3]
        
        broad_file_name = [filename for filename in os.listdir('.') if filename.endswith(".he")]
        broad_file = pd.read_fwf(broad_file_name[0], widths=field_lengths, header=None)
        
        gamma_he = np.array(broad_file[5])
        n_he = np.array(broad_file[8])
        J_lower = np.array(broad_file[14])
        
        J_max = max(J_lower)
        J_sorted = np.arange(int(J_max))
        
        gamma_he_avg, n_he_avg = (np.array([]) for _ in range(2))
        
        for i in range(int(J_max)):
            
            gamma_he_i = np.mean(gamma_he[np.where(J_lower == J_sorted[i])])
            n_he_i = np.mean(n_he[np.where(J_lower == J_sorted[i])])
            
            gamma_he_avg = np.append(gamma_he_avg, gamma_he_i)
            n_he_avg = np.append(n_he_avg, n_he_i)
        
        # Write He broadening file
        f_out = open('./He.broad','w')
            
        f_out.write('J | gamma_L_0 | n_L \n')
            
        for i in range(len(J_sorted)):
            f_out.write('%.1f %.4f %.3f \n' %(J_sorted[i], gamma_he_avg[i], n_he_avg[i]))
                
        f_out.close()
    
    elif (file_type == '.par'):
        
        field_lengths = [3,12,10,10,5,5,10,4,8,2,2,2,2,2,2,2,1,3,2,2,2,2,2,2,2,2,1,3,12,1,3,1,25,6,1,6]
    
        #trans_locations = re.findall(r'\d+', trans_file)
        #trans_1 = trans_locations[0]
        #trans_2 = trans_locations[1]
        
        trans_file = pd.read_fwf('./' + trans_file, widths=field_lengths, header=None)
        
        #iso = np.array(trans_file[1])
        nu_0 = np.array(trans_file[1])
        S_ref = np.array(trans_file[2])/X_iso[species]
        A = np.array(trans_file[3])
        gamma_L_0_air = np.array(trans_file[4])/1.01325   # Convert from cm^-1 / atm -> cm^-1 / bar
        E_lower = np.array(trans_file[6])
        n_L_air = np.array(trans_file[7])
        J_lower = np.array(trans_file[30])
        
        del trans_file
        
        # Just select transitions of primary imsotope
        #iso_condition = np.where(iso == 1)
        
        #nu_0 = nu_0[iso_condition]
        #S_ref = S_ref[iso_condition]
        #A = A[iso_condition]
        #gamma_L_0_air = gamma_L_0_air[iso_condition]
        #E_lower = E_lower[iso_condition]
        #n_L_air = n_L_air[iso_condition]
        #J_lower = J_lower[iso_condition]
        
        np.savetxt(prefix2 + linelist + '__' + file_number + '.trans',
                   np.transpose([nu_0, S_ref, E_lower, J_lower]), fmt='%.6f, %.3e, %.4f, %.1f')
          
        return J_lower, gamma_L_0_air, n_L_air
    
def process_VALD_file(species):
    
    trans_file = [filename for filename in os.listdir('.') if filename.endswith(".trans")]
     
    wl = []
    log_gf = []
    E_low = []   
    E_up = []    
    l_low = []
    l_up = []
    J_low = []
    J_up = []
    log_gamma_vdw = []

    f_in = open(trans_file[0], 'r')
    
    count = 0

    for line in f_in:
        
        count += 1
        
        if (count >= 3):
            
            if ((count+1)%4 == 0):
            
                line = line.strip()               
                line = line.split(',')
                
                # If at beginning of file footnotes, do not read further
                if (line[0] == '* oscillator strengths were NOT scaled by the solar isotopic ratios.'): break
                
                wl.append(float(line[1]))   # Convert wavelengths to um
                log_gf.append(float(line[2]))
                E_low.append(float(line[3]))
                J_low.append(float(line[4]))
                E_up.append(float(line[5]))
                J_up.append(float(line[6]))
                log_gamma_vdw.append(float(line[12]))
            
            elif ((count)%4 == 0):
                    
                # Only need orbital angular momentum QNs for alkalis
                if (species in ['Li', 'Na', 'K', 'Rb', 'Cs']):
                    
                    line = line.strip()               
                    line = line.split()
                        
                    # Orbital angular momentum quntum numbers
                    if   (line[2].endswith('s')): l_low.append(0)
                    elif (line[2].endswith('p')): l_low.append(1)
                    elif (line[2].endswith('d')): l_low.append(2)
                    elif (line[2].endswith('f')): l_low.append(3)
                    elif (line[2].endswith('g')): l_low.append(4)
                    else: print ("Error: above g orbital!")

            elif ((count-1)%4 == 0):
                    
                # Only need orbital angular momentum QNs for alkalis
                if (species in ['Li', 'Na', 'K', 'Rb', 'Cs']):
                    
                    line = line.strip()               
                    line = line.split()
                        
                    # Orbital angular momentum quntum numbers
                    if   (line[2].endswith('s')): l_up.append(0)
                    elif (line[2].endswith('p')): l_up.append(1)
                    elif (line[2].endswith('d')): l_up.append(2)
                    elif (line[2].endswith('f')): l_up.append(3)
                    elif (line[2].endswith('g')): l_up.append(4)
                    else: print ("Error: above g orbital!")                    
      
    f_in.close()
    
    nu = 1.0e4/np.array(wl)
    nu = nu[::-1]
    
    # Reverse array directions for increasing wavenumber
    wl = np.array(wl[::-1]) * 1.0e-3       # Convert nm to um
    log_gf = np.array(log_gf[::-1])
    E_low = np.array(E_low[::-1]) * 8065.547574991239  # Convert eV to cm^-1
    E_up = np.array(E_up[::-1]) * 8065.547574991239
    l_low = np.array(l_low[::-1])
    l_up = np.array(l_up[::-1])
    J_low = np.array(J_low[::-1])
    J_up = np.array(J_up[::-1])
    log_gamma_vdw = np.array(log_gamma_vdw[::-1])
    
    # Compute transition wavenumbers
    nu = 1.0e4/np.array(wl)
    
    # Compute gf factor
    gf = np.power(10.0, log_gf)
    
    # Open output file
    f_out = open(prefix2 + species + '_' + linelist + '.trans','w')
    
    f_out.write('nu_0 | gf | E_low | E_up | J_low | J_up | l_low | l_up | log_gamma_vdw \n')
    
    for i in range(len(nu)):
        f_out.write('%.6f %.6e %.6f %.6f %.1f %.1f %d %d %.6f \n' %(nu[i], gf[i], E_low[i], E_up[i],
                                                                    J_low[i], J_up[i], l_low[i], l_up[i],
                                                                    log_gamma_vdw[i]))
        
    f_out.close()
    
    return
    
        
if (linelist_type == 'EXOMOL'):

    process_Exomol_file(prefix, prefix2, '.broad', 'H2', 'dummy', 'dummy')
    process_Exomol_file(prefix, prefix2, '.broad', 'He', 'dummy', 'dummy')
    #process_Exomol_file(prefix, prefix2, '.states', 'dummy', 'dummy', 'dummy')
    process_Exomol_file(prefix, prefix2, '.pf', 'dummy', 'dummy', 'dummy')

elif (linelist_type == 'HITRAN'):
    
    process_HITRAN_file('.pf', chem_species, 'dummy', 0)

    #process_HITRAN_file('.h2', 'dummy', 'dummy', 0)
    #process_HITRAN_file('.he', 'dummy', 'dummy', 0)

    trans_files = [filename for filename in os.listdir('.') if filename.endswith(".par")]

    J_lower_all, gamma_air, n_air = (np.array([]) for _ in range(3))
    gamma_air_avg, n_air_avg = (np.array([]) for _ in range(2))

    for i in range(len(trans_files)):
        
        J_i, gamma_air_i, n_air_i = process_HITRAN_file('.par', chem_species, trans_files[i], str(i))
    
        J_lower_all = np.append(J_lower_all, J_i)
        gamma_air = np.append(gamma_air, gamma_air_i)
        n_air = np.append(n_air, n_air_i)
        
        print("File " + str(i+1) + " completed")
    
    # Compute average pressure broadening parameters as a function of J''
        
    J_max = max(J_lower_all)
    J_sorted = np.arange(int(J_max))
    
    for i in range(int(J_max)):
        
        gamma_air_i = np.mean(gamma_air[np.where(J_lower_all == J_sorted[i])])
        n_air_i = np.mean(n_air[np.where(J_lower_all == J_sorted[i])])
        
        gamma_air_avg = np.append(gamma_air_avg, gamma_air_i)
        n_air_avg = np.append(n_air_avg, n_air_i)
    
    # Write air broadening file
    f_out = open(prefix2 + 'air.broad','w')
        
    f_out.write('J | gamma_L_0 | n_L \n')
        
    for i in range(len(J_sorted)):
        f_out.write('%.1f %.4f %.3f \n' %(J_sorted[i], gamma_air_avg[i], n_air_avg[i]))
            
    f_out.close()

if (linelist_type == 'VALD'):
    
    process_VALD_file(chem_species)


"""







    

