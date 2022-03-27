# measure similarity
# Last changed : 2021.06.30
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr

import numpy as np
import pandas as pd
import pickle
import time
from numpy.linalg import norm
import math
import os


case = "a"
th = "1"
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
    print("==current => ", case, th, "|", iseuclid)
    start_time = time.time()

    feature_path =  "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/testdata_feature/"+case+th+".pkl"
    GT_path =  "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/mask_GT.pkl"
    test_data_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/fold"+th+"/test/dataset/"
    save_small_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/"
    save_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/"


    if iseuclid:
        save_small_path = save_small_path + "euclid/" + case + "/" + th
        save_path = save_path + "euclid/" + case + "/" + case + th + ".pkl"
    else:
        save_small_path = save_small_path + "cosine/" + case + "/" + th
        save_path = save_path + "cosine/" + case + "/" + case + th + ".pkl"

    if not(os.path.isdir(save_small_path)):
        os.mkdir(save_small_path)


    #load and read method
    with open(feature_path, "rb" ) as file:
        feature_data = pickle.load(file)

    with open(GT_path, "rb") as file:
        GT_data = pickle.load(file)

    
    file_load_time = time.time()
    period = file_load_time - start_time
    print("file load time : {}".format(period))

    df = pd.DataFrame()
    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]

    file_list = os.listdir(test_data_path)

    for i in range(len(file_list)):
        test_data = feature_data.loc[i]
        img_ID = test_data[1]
        feature = np.squeeze(test_data[0])
        caseNum = img_ID.split('_')[0]

        min_value = 0
        max_value = 0
        min_ID = ""
        max_ID = ""
        df_small = pd.DataFrame()
        test_GT = GT_data.loc[img_ID]

        first_iter = True

        for j in range(len(file_list)):
            db_img_ID = feature_data.loc[j][1]

            if db_img_ID.split('_')[0] == caseNum:
                continue

            compare_feature = np.squeeze(feature_data.loc[j][0])

            #distance
            distance = getDistance(feature, feature_data.loc[j][0])

            if first_iter == True:
                min_value = distance
                max_value = distance
                min_ID = db_img_ID
                max_ID = db_img_ID
                first_iter = False
                continue
            
            if distance > max_value:
                max_value = distance
                max_ID = db_img_ID

            if distance < min_value:
                min_value = distance
                min_ID = db_img_ID

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
        with open(save_small_path + "/" + img_ID  + ".pkl", "wb") as file:
            pickle.dump(df_small, file)

        min_GT = GT_data.loc[min_ID]
        resultsum = 0

        for j in range(6):
            if test_GT[organ_list[j]] == min_GT[organ_list[j]]:
                resultsum = resultsum + 1

        if i%100 == 0 : 
            print("current case : {}".format(caseNum))        
            print(min_value, max_value, min_ID, max_ID, resultsum)

        data = {"test_data_ID" : [img_ID], "max_img_ID" : [max_ID], "max_value" : [max_value], "min_img_ID" : [min_ID], "min_value" : [min_value], "sumOfGT" : [resultsum]}
        df_new = pd.DataFrame(data)
        df = df.append(df_new, sort=True).fillna(0)

    df = df.reset_index(drop=True)

    #save dataset
    with open(save_path, "wb") as file:
        pickle.dump(df, file)

    print(df)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            


if __name__ == "__main__":

    main()

    cases = ["a", "b", "f", "d", "e"]
    ths = ["1", "2"]

    for c in cases:
        case = c
        for t in ths:            
            th = t
            iseuclid = True
            main()

            iseuclid = False
            main()
