# original source : eval_linear.py
# Visualize feature map
# source : https://discuss.pytorch.org/t/visualize-feature-map/29597/2
# Last changed : 2021.07.18
# Editor : Kim su hyeon
# Email : kih629@gachon.ac.kr

import argparse
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import PIL
import pandas as pd
import pickle
import matplotlib.pyplot as plt


import resnet50 as resnet_models


logger = getLogger()

parser = argparse.ArgumentParser(description="Extractor feature using pretraind CNN by imageNet")


#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/home/sh/Desktop/env_swav/dataset/test_data",
                    help="path to dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")



#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="/home/sh/Downloads/swav_800ep_pretrain.pth.tar", type=str, help="path to pretrained weights")


parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")



#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")


# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def main():
    global args, best_acc
    args = parser.parse_args()

    # build data
    dataset = datasets.ImageFolder(os.path.join(args.data_path))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )

    dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        tr_normalize,
    ])

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Building data done")


    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)

    # model to gpu
    model = model.cuda()
    model.eval()


    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))
    else:
        print("No pretrained weights found => training with random weights")


    cudnn.benchmark = True


    model.eval()

    allFiles, _ = map(list, zip(*data_loader.dataset.samples))



    #make dataframe
    #index = file_name
    df = pd.DataFrame()


    for i, (img, target) in enumerate(data_loader):

        if i%100 == 0:
            print(i)

        # === show img ===
        # tf = transforms.ToPILImage()
        # img_png = tf(img.squeeze(0))
        # img_png.show()
        # plt.waitforbuttonpress(0)

        # move to gpu
        inp = img.cuda(non_blocking=True)

        model.conv1.register_forward_hook(get_activation('avgpool'))



        # forward
        with torch.no_grad():
            output = model(inp) #get feature

        print(output.shape)
        exit(0)

        # imgs = activation['avgpool'].squeeze().split(1, 0)
        # for img in imgs:
        #     plt.figure()
        #     plt.imshow(img.cpu().squeeze().numpy())
        #     plt.show()

        print(activation['avgpool'].shape)

        exit(0)

        act = activation['conv1'].squeeze()
        fig, axarr = plt.subplots(act.size(0))

        for idx in range(act.size(0)):
            axarr[idx].imshow(act[idx].cpu())

        exit(0)

       
        output = output.cpu().numpy()
        file_name = allFiles[i].split('/')[-1].split('.')[0] #get img ID

        data = {"idx" : [file_name], "feature" : [output]}

        df_new = pd.DataFrame(data)
        df = df.append(df_new, sort=True).fillna(0)
 

    save_path = "/home/sh/Desktop/env_swav/dataset/feature.pkl"

    df = df.reset_index(drop=True)


    #save dataset
    with open(save_path, "wb") as file:
        pickle.dump(df, file)

    print(df)



if __name__ == "__main__":
    main()
