#from GT masks, get and save information.
# Last changed : 2021.06.28
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr


import SimpleITK as sitk
import numpy as np
import pandas as pd
import copy
import pickle

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    return ct_scan




def main():
    data_path = "/home/sh/Desktop/env_swav/liver_dataset_old/MASK/"
    save_path =  "/home/sh/Desktop/env_swav/liver_dataset/dataset/mask_GT.pkl"

    organ_list = ["artery", "bone", "kidneys", "liver", "lungs", "spleen"]

    df = pd.DataFrame(columns = organ_list)


    for i in range(1, 21):
        src = data_path + str(i) + "/"

        for j in organ_list:
            input_src = src + j + "/" + j + ".mhd"
            np_img = load_itk(input_src)
            
            for k in range(np_img.shape[0]):
                imgID = str(i) +"_image_" + str(k)

                if(j == "artery"):
                    df_new = pd.DataFrame(index=[imgID])
                    df = df.append(df_new, sort=True).fillna(0)

                unique = np.unique(np_img[k])

                if(len(unique) > 1):
                    df.loc[[imgID], j] = 1
                


    df = df[organ_list]
    print(df)

    #save dataset
    with open(save_path, "wb") as file:
        pickle.dump(df, file)


    #load and read method
    with open(save_path, "rb" ) as file:
        loaded_data = pickle.load(file)
    
    print(loaded_data)


if __name__ == "__main__":
    main()
