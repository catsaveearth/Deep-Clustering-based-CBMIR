# date : 2021.09.14
# show data


import numpy as np
import pandas as pd
import pickle
import time
from numpy.linalg import norm
import math
import os

fold = "2"

test_data_path = "/home/sh/Desktop/liver_application_testset/fold" + fold + "/"
data_feature = "/home/sh/Desktop/liver_application_testset/result/fold" + fold + "/ver"


file_list = os.listdir(test_data_path)

for i in file_list:
    name = i[:-4]
    print(name, "===========euclid")

    for j in range(1, 5):
        path = data_feature + str(j) + "/"
        data_list = os.listdir(path)

        for k in data_list:
            if name.split("_")[0] == k.split("_")[0] and k.split("_")[3] == "Euclid":
                with open(path + k, "rb") as file:
                    feature_data = pickle.load(file)
                    feature_data = feature_data.sort_values(by=['distance'], axis=0, ascending=True)
                    print(feature_data.head(1))



for i in file_list:
    name = i[:-4]
    print("\n", name, "===========cosine")

    for j in range(1, 5):
        path = data_feature + str(j) + "/"
        data_list = os.listdir(path)

        for k in data_list:
            if name.split("_")[0] == k.split("_")[0] and k.split("_")[3] == "Cosine":
                with open(path + k, "rb") as file:
                    feature_data = pickle.load(file)
                    feature_data = feature_data.sort_values(by=['distance'], axis=0, ascending=True)
                    print(feature_data.head(1))
