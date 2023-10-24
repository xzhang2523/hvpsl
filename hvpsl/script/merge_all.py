import pandas as pd
import os
import numpy as np
import csv


all_problem_name = ['zdt1', 'zdt2', 'vlmop1','vlmop2', 'RE21', 'RE24', 'lqr2', 'RE37', 'lqr3']
# all_problem_name = ['adult', 'compass']


if __name__ == '__main__':
    # Firstly, calculate the average
    all_data = [0] * len(all_problem_name)
    all_data_std = [0] * len(all_problem_name)
    
    for idx, problem_name in enumerate(all_problem_name):
        folder_prefix = os.path.join(os.path.dirname( os.path.dirname(os.path.abspath(__file__)) ), 'output', problem_name)
        print()

        csv_name = os.path.join(folder_prefix, 'all.csv')
        res = pd.read_csv(csv_name)
        num_col = res.values.shape[-1]
        rr = res.values
        data = rr[:,1:]
        all_data[idx] = data
        
        
        csv_name_std = os.path.join(folder_prefix, 'all_std.csv')
        res_std = pd.read_csv(csv_name_std)
        # num_col = res.values.shape[-1]
        rr_std = res_std.values
        data_std = rr_std[:,1:]
        all_data_std[idx] = data_std
        


    all_data = [0] * len(all_problem_name)
    all_data_std = [0] * len(all_problem_name)
    
    for idx, problem_name in enumerate(all_problem_name):
        folder_prefix = os.path.join(os.path.dirname( os.path.dirname(os.path.abspath(__file__)) ), 'output', problem_name)

        csv_name = os.path.join(folder_prefix, 'all.csv')
        res = pd.read_csv(csv_name)
        num_col = res.values.shape[-1]
        rr = res.values
        
        csv_name_std = os.path.join(folder_prefix, 'all_std.csv')
        res_std = pd.read_csv(csv_name_std)
        num_col = res.values.shape[-1]
        rr_std = res_std.values
        
        for i in range(len(rr)):
            rr[i,0] = 'PSL-' + rr[i,0]
            
        
        for i in [1,2,3]:
            if i==1 or i==2:
                ccol = rr[:,i]
                ww = np.argwhere(ccol == np.max(ccol))
                for j in ww:
                    rr[j[0]][i] = '\\textbf{' + str(rr[j[0]][i]) + '}'

        
        # ress = np.concatenate((rr, np.zeros((1,5))), 0)
        all_data[idx] = rr
        all_data_std[idx] = rr_std
        # print()
        
    all_data = np.concatenate(all_data, 0)
    all_data_std = np.concatenate(all_data_std, 0)


    all_data_prime = np.zeros((15,13), dtype=object)
    all_data_prime[:,0] = all_data[:15,0]
    
    all_data_prime_std = np.zeros((15,13), dtype=object)
    all_data_prime_std[:,0] = all_data[:15,0]
    

    for i in range(15):
        for j in range(12):
            (m,n) = (int(i/5), int(j/4))
            (idx_m,idx_n) = (i%5, j%4)
            
            idx = (3*m+n)*5
            all_data_prime[i,j+1] = all_data[idx+idx_m, idx_n+1]
            all_data_prime_std[i,j+1] = all_data_std[idx+idx_m, idx_n+1]



    csv_file_name = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'script', 'synetic.csv')

    with open(csv_file_name, 'w', newline='', encoding='UTF-8') as csvfile:
        spamwriter = csv.writer(csvfile)
        one_row = ['Method', 'HV', 'Range', 'Sparsity', 'Time'] + ['HV', 'Range', 'Sparsity', 'Time'] + ['HV', 'Range', 'Sparsity', 'Time']
        print(one_row)
        spamwriter.writerow(one_row)
        for elem in all_data_prime:
            spamwriter.writerow(elem)
    print('saved in:{}'.format(csv_file_name))


    csv_file_name = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'script', 'synetic_std.csv')
    with open(csv_file_name, 'w', newline='', encoding='UTF-8') as csvfile:
        spamwriter = csv.writer(csvfile)
        one_row = ['Method', 'HV', 'Range', 'Sparsity', 'Time'] + ['HV', 'Range', 'Sparsity', 'Time'] + ['HV', 'Range', 'Sparsity', 'Time']
        print(one_row)
        spamwriter.writerow(one_row)
        for elem in all_data_prime_std:
            spamwriter.writerow(elem)
    print('saved in:{}'.format(csv_file_name))