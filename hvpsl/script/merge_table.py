import os
import csv
import pandas as pd 
import numpy as np
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')
    parser.add_argument('--problem-name', type=str, default='zdt2')

    args = parser.parse_args()
    problem_name = args.problem_name

    folder_prefix = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    folder_prefix = os.path.join(folder_prefix, 'output', args.problem_name )
    # folder_prefix = 'C:\\Users\\xzhang2523\\Desktop\\IJCAI_submit\\HV_PSL\\Figures\\exp2\\SGD\\{}'.format(problem_name)



    decompose_array = ['epo','ls','tche','hv1', 'hv2']

    all_data = np.zeros((len(decompose_array), 4))
    all_data_std = np.zeros((len(decompose_array), 4))

    for idx, decompose in enumerate(decompose_array):
        try:
            seed_num = 3
            res_arr = [0] * seed_num
            for seed in range(seed_num):
                folder_name = os.path.join(folder_prefix, 'seed_{}'.format(seed))
                csv_name = os.path.join(folder_name, '{}.csv'.format(decompose))
                res_arr[seed] = pd.read_csv(csv_name).values[0]

            res = np.round(np.mean(np.array(res_arr),0), 2)
            res_std = np.round(np.std(np.array(res_arr),0), 2)


            all_data[idx,:] = res
            all_data_std[idx,:] = res_std

        except:
            print('{} {} not implemented'.format(args.problem_name, decompose))



    mtd_dict = {
        'hv1' : 'HV1',
        'hv2' : 'HV2',
        'ls' : 'LS',
        'epo' : 'EPO',
        'tche' : 'Tche',
    }


    csv_file_name = os.path.join(folder_prefix, '{}.csv'.format('all'))
    csv_file_name_std = os.path.join(folder_prefix, '{}.csv'.format('all_std'))


    with open(csv_file_name, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        one_row = ['Method', 'HV', 'Range', 'Sparsity', 'Time']
        print(one_row)
        spamwriter.writerow(one_row)

        for idx, decompose in enumerate(decompose_array):
            one_row = [mtd_dict[decompose], '{}'.format(all_data[idx,0]), '{}'.format(all_data[idx,1]), '{:.2f}'.format(all_data[idx,2]), '{}'.format(all_data[idx,3])]
            print(one_row)
            spamwriter.writerow(one_row)
    print('saved in {}'.format(csv_file_name))


    with open(csv_file_name_std, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        one_row = ['Method', 'HV', 'Range', 'Sparsity', 'Time']
        print(one_row)
        spamwriter.writerow(one_row)

        for idx, decompose in enumerate(decompose_array):
            one_row = [mtd_dict[decompose], '{}'.format(all_data_std[idx,0]), '{}'.format(all_data_std[idx,1]), '{:.2f}'.format(all_data_std[idx,2]), '{}'.format(all_data_std[idx,3])]
            print(one_row)
            spamwriter.writerow(one_row)
    print('saved in {}'.format(csv_file_name_std))

    
    
    
    



