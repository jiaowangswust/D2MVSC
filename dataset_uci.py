from __future__ import print_function
import torch.utils.data as data
import scipy.io
import numpy as np


class UCI(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.train_num = int(2000)
        # data_0 = scipy.io.loadmat('rand/yaleA_3view.mat')
        # data_dict = dict(data_0)
        # data_1 = data_dict['X']
        #
        # self.data1 = data_1[0][0].astype(np.float32)
        # self.data2 = data_1[1][0].astype(np.float32)
        # self.data3 = data_1[2][0].astype(np.float32)
        data_0 = scipy.io.loadmat('uci-digit_m.mat')
        data_dict = dict(data_0)
        data_1 = data_dict['mfeat_fac']
        data_2 = data_dict['mfeat_fou']
        data_3 = data_dict['mfeat_kar']

        self.data1 = data_1.astype(np.float32)
        self.data2 = data_2.astype(np.float32)
        self.data3 = data_3.astype(np.float32)
        # self.data3 = data_1[0][2].astype(np.float32)
        print(self.data1.shape)
        print(self.data2.shape)
        print(self.data3.shape)
        # print(self.data3.shape)

    def __getitem__(self, index):
        # img_train1, img_train2, img_train3 = self.data1[index, :], self.data2[index, :], self.data3[index, :]
        # return img_train1, img_train2, img_train3

        img_train1, img_train2, img_train3 = self.data1[index, :], self.data2[index, :], self.data3[index, :]
        return img_train1, img_train2,img_train3

    def __len__(self):
        return self.train_num




