# compare with GT for period?
# Last changed : 2021.07.01
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr

import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt



def peroid(i):
    if i<140:
        return 0
    elif i<150:
        return 1
    elif i < 160:
        return 2
    elif i<170:
        return 3
    else:
        return 4



def main():
    start_time = time.time()
    compare_path =  "/home/sh/Desktop/env_swav/dataset/compare_feature_euclid_ft.pkl"
    GT_path =  "/home/sh/Desktop/env_swav/dataset/MASK/mask_GT.pkl"

    #load and read method
    with open(compare_path, "rb" ) as file:
        feature_data = pickle.load(file)

    #load and read method
    with open(GT_path, "rb" ) as file:
        GT_data = pickle.load(file)


    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]
    organ_check = [[0 for j in range(6)] for i in range(5)]
    real_organ_check = [[0 for j in range(6)] for i in range(5)]


    for i in range(0, 2823):
        data_info = feature_data.loc[i]

        test_data_ID = data_info["test_data_ID"]
        min_ID = data_info["min_img_ID"]
        mv = peroid(data_info["min_value"])

        I_GT = GT_data.loc[test_data_ID]
        J_GT = GT_data.loc[min_ID]


        for j in range(6):
            if I_GT[organ_list[j]] == 1:
                real_organ_check[mv][j] = real_organ_check[mv][j] + 1

            if I_GT[organ_list[j]] == 1 and J_GT[organ_list[j]] == 1:
                organ_check[mv][j] = organ_check[mv][j] + 1


    for i in real_organ_check:
        print(i)


    # for i in organ_check:
    #     print(i)



if __name__ == "__main__":
    main()
