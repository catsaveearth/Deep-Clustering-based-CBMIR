# show dataframe describe
# Last changed : 2021.06.30
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr


import numpy as np
import pandas as pd
import pickle


def main():
    data_path =  "/home/sh/Desktop/env_swav/liver_dataset/2fold_result/cosine/e/e1.pkl"


    #load and read method
    with open(data_path, "rb" ) as file:
        loaded_data = pickle.load(file)

    print(loaded_data)

    print(loaded_data.describe())


if __name__ == "__main__":
    main()
