# compare with GT
# Last changed : 2021.07.01
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr

import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    start_time = time.time()
    compare_path =  "/home/sh/Desktop/env_swav/dataset/compare_feature_cosine_ft.pkl"
    GT_path =  "/home/sh/Desktop/env_swav/dataset/MASK/mask_GT.pkl"

    #load and read method
    with open(compare_path, "rb" ) as file:
        feature_data = pickle.load(file)

    #load and read method
    with open(GT_path, "rb" ) as file:
        GT_data = pickle.load(file)


    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]
    organ_check = [0 for j in range(6)]
    check_case = [[0 for j in range(20)] for i in range(20)]


    for i in range(0, 2823):
        data_info = feature_data.loc[i]

        test_data_ID = data_info["test_data_ID"]
        min_ID = data_info["min_img_ID"]

        I_GT = GT_data.loc[test_data_ID]
        J_GT = GT_data.loc[min_ID]

        for j in range(6):
            if I_GT[organ_list[j]] == 1 and J_GT[organ_list[j]] == 1:
                organ_check[j] = organ_check[j] + 1


        caseNum_I = test_data_ID.split('_')[0]
        caseNum_J = min_ID.split('_')[0]
        check_case[int(caseNum_I) - 1][int(caseNum_J) -1] = check_case[int(caseNum_I) - 1][int(caseNum_J) -1] + 1

    print(organ_check)
    
    ax = sns.heatmap(check_case, cmap="YlGnBu")
    plt.show()

    # plt.imshow(check_case, cmap='hot', interpolation='nearest')
    # plt.xticks(np.arange(0, 20, 1))
    # plt.yticks(np.arange(0, 20, 1))

    # plt.show()




if __name__ == "__main__":
    main()
