# check GT elements
# Last changed : 2021.07.01
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr


import numpy as np
import pandas as pd
import pickle
import os

th = "2"

def main():
    GT_path =  "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/mask_GT.pkl"
    test_data_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/fold"+th+"/test/dataset/"

    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]

    #load and read method
    with open(GT_path, "rb" ) as file:
        GT_data = pickle.load(file)

    organ_check = [0 for j in range(6)]

    file_list = os.listdir(test_data_path)


    for j in file_list:
        data_info = GT_data.loc[j[:-4]]

        for i in range(6):
            if data_info[organ_list[i]] == 1:
                organ_check[i] = organ_check[i] + 1

    print(organ_check)


if __name__ == "__main__":
    main()
