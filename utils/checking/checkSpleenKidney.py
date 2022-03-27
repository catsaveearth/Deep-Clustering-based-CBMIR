# check GT elements
# Last changed : 2021.07.28
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import pprint as pp


def ranges(i):
    if i<= 0.35:
        return 0
    elif i<=0.4:
        return 1
    elif i<= 0.45:
        return 2
    elif i<= 0.5:
        return 3
    elif i<= 0.55:
        return 4
    else:
        return 5


def main():
    feature_compare_path =  "/home/sh/Desktop/env_swav/dataset/compare_feature_notFT/compare_feature_euclid_ver2.pkl"
    GT_path =  "/home/sh/Desktop/env_swav/dataset/MASK/mask_GT.pkl"


    #load and read method
    with open(feature_compare_path, "rb" ) as file:
        feature_data = pickle.load(file)
    feature_data = feature_data.sort_values(by=['min_value'], axis=0, ascending=False)
    print(feature_data.head(5))
    exit(0)


    with open(GT_path, "rb" ) as file:
        GT_data = pickle.load(file)


    sumOfGT = [[0 for j in range(4)] for i in range(6)]
    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]
    range_check = [0 for j in range(6)]


    for i in range(2823):
        resultsum = 0
        data_info = feature_data.iloc[i]
        testdata = data_info['test_data_ID']
        mindata = data_info['min_img_ID']
        minvalue = data_info['min_value']

        test_GT = GT_data.loc[testdata]
        min_GT = GT_data.loc[mindata]

        for j in range(6):
            if test_GT[organ_list[j]] == min_GT[organ_list[j]]:
                resultsum = resultsum + 1

        range_result = ranges(minvalue)


        sumOfGT[range_result][resultsum] = sumOfGT[range_result][resultsum] + 1
        range_check[range_result] = range_check[range_result] + 1

    print("\n\n=====confusion matrix=====")
    pp.pprint(sumOfGT)
    pp.pprint(range_check)

if __name__ == "__main__":
    main()
