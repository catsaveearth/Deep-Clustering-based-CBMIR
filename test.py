# date : 2021.09.13
# for CBMIR application

import numpy as np
import pandas as pd
import pickle
import time
from numpy.linalg import norm
import math
import os

fold = "1"
otherfold = "2"
iseuclid = False
disttype = "none"
feature_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/testdata_feature/a" + fold + ".pkl"
feature_path2 = "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/testdata_feature/a" + otherfold + ".pkl"

GT_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/mask_GT.pkl"
feature_data = ""
feature_data2 = ""
GT_data = ""

def cosine_similarity(a, b):
    dot = a * b
    return dot.sum()/(norm(a)*norm(b))


def getDistance(a, b):
    #type => T : euclid / F : cosine
    if iseuclid:
        #Euclidean
        distance = a - b
        distance = distance * distance
        distance = np.sqrt(distance.sum())
    else:
        #cosine
        similarity = cosine_similarity(a, b)
        distance = 1-similarity
    
    return distance


def main():
    print("start")
    test_data_path = "/home/sh/Desktop/liver_application_testset/fold" + fold + "/"
    compare_data_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/fold" + fold + "/test/dataset/"
    compare_data_path2 = "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/fold" + fold + "/train/dataset/"
    #compare_data_path = "/home/sh/Desktop/env_swav/liver_dataset_old/test_data/all_raw_data/"
    save_small_path = "/home/sh/Desktop/liver_application_testset/result/fold" + fold + "/"

    if iseuclid:
        disttype = "Euclid"
    else:
        disttype = "Cosine"


    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]

    file_list = os.listdir(test_data_path)
    compare_list = os.listdir(compare_data_path)
    compare_list2 = os.listdir(compare_data_path2)



    for i in range(len(feature_data)):
        test_data = feature_data.loc[i]
        img_ID = test_data[1]

        if img_ID + ".png" in file_list:
            print(img_ID)
        else:
            continue


        feature = np.squeeze(test_data[0])
        caseNum = img_ID.split('_')[0]

        df_small = pd.DataFrame()
        test_GT = GT_data.loc[img_ID]


        for j in range(len(compare_list)):
            db_img_ID = feature_data.loc[j][1]

            if db_img_ID.split('_')[0] == caseNum:
                continue

            #distance
            distance = getDistance(feature, feature_data.loc[j][0])

            compare_GT = GT_data.loc[db_img_ID]
            rsum = 0

            for j in range(6):
                if test_GT[organ_list[j]] == compare_GT[organ_list[j]]:
                    rsum = rsum + 1

            data_small = {"comp_img_ID" : [db_img_ID], "distance" : [distance], "sumOfGT" : [rsum]}
            df_new_small = pd.DataFrame(data_small)
            df_small = df_small.append(df_new_small, sort=True).fillna(0)


        for j in range(len(compare_list2)):
            db_img_ID = feature_data2.loc[j][1]

            if db_img_ID.split('_')[0] == caseNum:
                continue

            #distance
            distance = getDistance(feature, feature_data2.loc[j][0])

            compare_GT = GT_data.loc[db_img_ID]
            rsum = 0

            for j in range(6):
                if test_GT[organ_list[j]] == compare_GT[organ_list[j]]:
                    rsum = rsum + 1

            data_small = {"comp_img_ID" : [db_img_ID], "distance" : [distance], "sumOfGT" : [rsum]}
            df_new_small = pd.DataFrame(data_small)
            df_small = df_small.append(df_new_small, sort=True).fillna(0)


        df_small = df_small.reset_index(drop=True)

        #save dataset
        with open(save_small_path + img_ID + "_" + disttype + "_ver4" + ".pkl", "wb") as file:
            pickle.dump(df_small, file)


if __name__ == "__main__":

    #load and read method
    with open(feature_path, "rb" ) as file:
        feature_data = pickle.load(file)

    with open(feature_path2, "rb" ) as file:
        feature_data2 = pickle.load(file)

    with open(GT_path, "rb") as file:
        GT_data = pickle.load(file)

    iseuclid = True
    main()

    iseuclid = False
    main()