# compare with GT for confusion matrix per label
# Last changed : 2021.07.19
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr

import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pp
import os

case = "a"
th = "1"

def main():
    start_time = time.time()
    compare_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/cosine/" + case + "/" + case + th + ".pkl"
    GT_path =  "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/mask_GT.pkl"
    test_data_path = "/home/sh/Desktop/env_swav/liver_dataset/2fold_dataset/fold"+th+"/test/dataset/"

    #load and read method
    with open(compare_path, "rb" ) as file:
        feature_data = pickle.load(file)

    #load and read method
    with open(GT_path, "rb" ) as file:
        GT_data = pickle.load(file)


    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]
    organ_check = [0 for j in range(6)]
    check_case = [[0 for j in range(20)] for i in range(20)]
    confusion_matrix_per_label = [[0 for j in range(4)] for i in range(6)]
    # TP, TN, FP, FN


    file_list = os.listdir(test_data_path)

    for i in range(len(file_list)):
        data_info = feature_data.loc[i]

        test_data_ID = data_info["test_data_ID"]
        min_ID = data_info["min_img_ID"]


        I_GT = GT_data.loc[test_data_ID]
        J_GT = GT_data.loc[min_ID]

        for j in range(6):
            if I_GT[organ_list[j]] == 1 and J_GT[organ_list[j]] == 1:
                organ_check[j] = organ_check[j] + 1
                confusion_matrix_per_label[j][0] = confusion_matrix_per_label[j][0] + 1  #TP

            elif I_GT[organ_list[j]] == 0 and J_GT[organ_list[j]] == 1:
                confusion_matrix_per_label[j][1] = confusion_matrix_per_label[j][1] + 1  #TN

            elif I_GT[organ_list[j]] == 1 and J_GT[organ_list[j]] == 0:
                confusion_matrix_per_label[j][2] = confusion_matrix_per_label[j][2] + 1  #FP

            elif I_GT[organ_list[j]] == 0 and J_GT[organ_list[j]] == 0:
                confusion_matrix_per_label[j][3] = confusion_matrix_per_label[j][3] + 1  #FN


        caseNum_I = test_data_ID.split('_')[0]
        caseNum_J = min_ID.split('_')[0]
        check_case[int(caseNum_I) - 1][int(caseNum_J) -1] = check_case[int(caseNum_I) - 1][int(caseNum_J) -1] + 1

    print("\n>>>>" + case + th + "<<<<")
    #print("organ_check => ", organ_check)

    # print("\n\n=====confusion matrix=====")
    # pp.pprint(confusion_matrix_per_label)

    for k, ctx in enumerate(confusion_matrix_per_label):
        # TP, TN, FP, FN
        precision = float(ctx[0]) / float(ctx[0] + ctx[1])
        recall = float(ctx[0])/ float(ctx[0] + ctx[2])
        f1score = 2 * precision * recall / (precision + recall)
        print(organ_list[k], "=> F1 scpre : ", f1score)
        #print(organ_list[k], " => precision : ", precision, "| recall : ", recall)



    
    # ax = sns.heatmap(check_case, cmap="YlGnBu")
    # plt.show()

    # plt.imshow(check_case, cmap='hot', interpolation='nearest')
    # plt.xticks(np.arange(0, 20, 1))
    # plt.yticks(np.arange(0, 20, 1))

    # plt.show()




if __name__ == "__main__":
    cases = ["a", "b", "d", "e", "f"]
    ths = ["1", "2"]

    for c in cases:
        case = c

        for t in ths:          
            th = t
            main()

        
        print("\n\n\n")