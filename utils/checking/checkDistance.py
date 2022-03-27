# check GT elements
# Last changed : 2021.07.01
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

case = "a"
th = "1"
img_ID = "9_image_54"

def main():
    save_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/cosine/" + case + "/" + case + th + ".pkl"
    #single_img_save_path =  "/home/sh/Desktop/env_swav/liver_dataset/result/" + "euclid/" + case + "/" + th + "/" + img_ID  + ".pkl"


    #load and read method
    with open(save_path, "rb" ) as file:
        feature_data = pickle.load(file)

    feature_data = feature_data.sort_values(by=['min_value'], axis=0, ascending=True)
    # print(feature_data.head(5))
    # return



    #load and read method
    with open(single_img_save_path, "rb" ) as file:
        single_feature_data = pickle.load(file)
 
    single_feature_data = single_feature_data.sort_values(by=['distance'], axis=0, ascending=True)


    ndarray = img.imread("/home/sh/Desktop/env_swav/dataset/test_data/all_raw_data/" + title + ".png")
    ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
    plt.title("{}".format(title))
    plt.imshow(ndarray)
    plt.show()

    for i in range(0, 1000, 200):
        data_info = single_feature_data.iloc[i]
        compdata = data_info['comp_img_ID']

        plt.subplot(121)
        ndarray = img.imread("/home/sh/Desktop/env_swav/dataset/test_data/all_raw_data/" + compdata + ".png")
        ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
        plt.title("{} | {} | {}".format(i, data_info['distance'], data_info['sumOfGT']))
        plt.imshow(ndarray)


        data_info = single_feature_data.iloc[i+100]
        compdata = data_info['comp_img_ID']

        plt.subplot(122)
        ndarray = img.imread("/home/sh/Desktop/env_swav/dataset/test_data/all_raw_data/" + compdata + ".png")
        ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
        plt.title("{} | {} | {}".format(i + 100, data_info['distance'], data_info['sumOfGT']))
        plt.imshow(ndarray)
        plt.show()



if __name__ == "__main__":
    cases = ["a", "b", "d", "e", "f"]
    ths = ["1", "2"]

    for c in cases:
        case = c
        for t in ths:
            th = t
            print("===", case, th, "===")
            main()

