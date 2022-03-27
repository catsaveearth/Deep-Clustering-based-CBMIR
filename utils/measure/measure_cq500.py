# measure similarity
# Last changed : 2021.08.24
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr

import numpy as np
import pandas as pd
import pickle
import time
from numpy.linalg import norm
import math
import os


iseuclid = False

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
    start_time = time.time()

    feature_path =  "/home/sh/Desktop/env_swav/CQ500/dataset50_result/feature/"
    GT_path =  "/home/sh/Desktop/env_swav/CQ500/cq500_GT.pkl"
    save_small_path = "/home/sh/Desktop/env_swav/CQ500/dataset50_result/"
    save_path = "/home/sh/Desktop/env_swav/CQ500/dataset50_result/"


    if iseuclid:
        save_small_path = save_small_path + "euclid/"
        save_path = save_path + "euclid/total.pkl"
    else:
        save_small_path = save_small_path + "cosine/"
        save_path = save_path + "cosine/total.pkl"


    features = [0 for j in range(50)]

    if os.path.isdir(feature_path):
        feature_list = os.listdir(feature_path)

    #load and read method
    for i, name in enumerate(feature_list):
        with open(feature_path + name, "rb" ) as file:
            features[i] = (name[0:-4], pickle.load(file))
        continue


    with open(GT_path, "rb") as file:
        GT_data = pickle.load(file)

    
    file_load_time = time.time()
    period = file_load_time - start_time
    print("file load time : {}".format(period))


    df = pd.DataFrame()
    disease_list = ["ICH", "IPH", "IVH", "SDH", "EDH", "SAH", "BleedLocation-Left", "BleedLocation-Right", "ChronicBleed", "Fracture", "CalvarialFracture", "OtherFracture", "MassEffect", "MidlineShift"]

    
    for i in range(len(feature_list)):
        test_data = features[i]
        img_ID = test_data[0] #image number
        feature = test_data[1] #10 image feature per 1 folder
        print("1 : ", img_ID)


        min_value = 0
        max_value = 0
        min_ID = ""
        max_ID = ""
        df_small = pd.DataFrame()
        test_GT = GT_data.loc[img_ID]

        first_iter = True

        for j in range(len(feature_list)):

            compare_data = features[j]
            compare_img_ID = compare_data[0]

            if compare_img_ID == img_ID: #same folder? => skip
                continue


            compare_feature = compare_data[1] #10 image feature per 1 folder

            distance_list = [0 for k in range(10)]

            #distance for each 10 image
            for k in range(10):
                this_feature = np.squeeze(feature.loc[k][0])
                other_feature = np.squeeze(compare_feature.loc[k][0])
                distance_list[k] = getDistance(this_feature, other_feature)

            distance = sum(distance_list)

            if first_iter == True:
                min_value = distance
                max_value = distance
                min_ID = compare_img_ID
                max_ID = compare_img_ID
                first_iter = False
                continue
            
            if distance > max_value:
                max_value = distance
                max_ID = compare_img_ID

            if distance < min_value:
                min_value = distance
                min_ID = compare_img_ID

            compare_GT = GT_data.loc[compare_img_ID]
            rsum = 0

            for k in range(14):
                if test_GT[disease_list[k]] == compare_GT[disease_list[k]]:
                    rsum = rsum + 1

            data_small = {"comp_img_ID" : [compare_img_ID], "distance" : [distance], "sumOfGT" : [rsum]}
            df_new_small = pd.DataFrame(data_small)
            df_small = df_small.append(df_new_small, sort=True).fillna(0)


        df_small = df_small.reset_index(drop=True)

        # #save dataset
        # with open(save_small_path + img_ID  + ".pkl", "wb") as file:
        #     pickle.dump(df_small, file)

        min_GT = GT_data.loc[min_ID]
        resultsum = 0

        for j in range(14):
            if test_GT[disease_list[j]] == min_GT[disease_list[j]]:
                resultsum = resultsum + 1


        data = {"test_data_ID" : [img_ID], "max_img_ID" : [max_ID], "max_value" : [max_value], "min_img_ID" : [min_ID], "min_value" : [min_value], "sumOfGT" : [resultsum]}
        df_new = pd.DataFrame(data)
        df = df.append(df_new, sort=True).fillna(0)

    df = df.reset_index(drop=True)

    #save dataset
    with open(save_path, "wb") as file:
        pickle.dump(df, file)

    print(df)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            


if __name__ == "__main__":
    iseuclid = False
    main()

    iseuclid = True
    main()