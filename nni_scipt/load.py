
import csv
import pandas
from matplotlib import pyplot as plt






if __name__ == '__main__':
    file_name = 'D:\\pycharm_project\\hvpsl\\hvpsl\\res.csv'
    df = pandas.read_csv(file_name)

    nadir_eps = df.loc[:, 'nadir_eps'].values
    reward = df.loc[:, 'reward'].values


    plt.scatter(nadir_eps, reward)
    plt.show()

    # ['nadir_eps']

    print()
    # with open('data.csv', 'w',