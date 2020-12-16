# import os
import torch.utils.data as data
import numpy as np
from utils import YUVread
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision import transforms
import cv2
import random
# import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lmdb
import util
from PIL import Image
from torchvision.transforms import ToTensor
import yaml


class DataYUV(data.Dataset):
    def __init__(self, rec_path='../data/videos/BasketballDrive_1920x1080_50_000to049_QP22_IP_rec.yuv', label_path='../data/videos/BasketballDrive_1920x1080_50_000to049.yuv', totalFrames=50, nFrames=7, width=1920, heigh=1080, width_cut=480, heigh_cut=270):
        super(DataYUV, self).__init__()
        self.len = width*heigh*totalFrames//width_cut//heigh_cut
        self.nFrames = nFrames
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = heigh
        self.width_cut = width_cut
        self.heigh_cut = heigh_cut
        self.rec_file = open(rec_path, 'rb')
        self.label_file = open(label_path, 'rb')

    def __getitem__(self, idx):
        label_frame_num = idx//(self.width*self.heigh//self.width_cut//self.heigh_cut)
        middle_num = self.nFrames//2   # 3

        ### read multi input data
        # top 3 frames
        if label_frame_num < middle_num:
            # number of frames to be read
            nFrames = middle_num + 1 + (label_frame_num % middle_num)
            y, _, _ = YUVread(self.rec_file, [self.heigh, self.width], frame_num=nFrames, start_frame=0)
            y_fill = y[0].copy().reshape(-1)
            for _ in range(self.nFrames-nFrames):
                # print(y.shape, y[0].shape)
                # copy top 1 to fill 
                y = np.concatenate([y_fill, y.reshape(-1)])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # last 3 frames
        elif label_frame_num > self.totalFrames-middle_num-1:
            nFrames_2 = self.nFrames - (label_frame_num % (self.totalFrames-middle_num-1))
            start_frame = min(label_frame_num - middle_num, self.totalFrames-middle_num-1)
            y, _, _ = YUVread(self.rec_file, [self.heigh, self.width], frame_num=nFrames_2, start_frame=start_frame)
            y_fill = y[-1].copy().reshape(-1)
            for _ in range(self.nFrames-nFrames_2):
                y = np.concatenate([y.reshape(-1), y_fill])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # middle frames
        else:
            # number of frames to be read
            # nFrames = self.nFrames
            start_frame = label_frame_num - middle_num
            y, _, _ = YUVread(self.rec_file, [self.heigh, self.width], frame_num=self.nFrames, start_frame=start_frame)
            

        ### read label data
        label, _, _ = YUVread(self.label_file, [self.heigh, self.width], frame_num=1, start_frame=label_frame_num)
        # print(label.size)
        # label = np.array(label)
        # print(type(label))

        ### cut
        cut_num = idx % (self.width*self.heigh//self.width_cut//self.heigh_cut)
        w, h = cut_num // (self.width//self.width_cut), cut_num % (self.width//self.width_cut)
        # print(w, h)


        # input_ = y[middle_num, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)


        input_ = y[middle_num, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)


        label_ = label[:, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        
        neighbor = y[[2,4,1,5,0,6], h*self.heigh_cut:(h+1)*self.heigh_cut, w*self.width_cut:(w+1)*self.width_cut].reshape(6, -1, self.heigh_cut, self.width_cut)


        return torch.from_numpy(input_/255).float(), torch.from_numpy(neighbor/255).float(), torch.from_numpy(label_/255).float()

    def __len__(self):
        return self.len


class DataYUV_v1(data.Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=7, width=1920, heigh=1080, width_cut=480, heigh_cut=270):
        super(DataYUV_v1, self).__init__()
        self.nFrames = nFrames
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = heigh
        self.width_cut = width_cut
        self.heigh_cut = heigh_cut
        self.rec_file = rec_y
        self.label_file = label_y
        self.len = (width//width_cut)*(heigh//heigh_cut)*totalFrames



    def __getitem__(self, idx):
        # label_frame_num = idx//(self.width*self.heigh//self.width_cut//self.heigh_cut)
        label_frame_num = idx // ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        middle_num = self.nFrames//2   # 3

        ### read multi input data
        # top 3 frames
        if label_frame_num < middle_num:
            # number of frames to be read
            nFrames = middle_num + 1 + (label_frame_num % middle_num)
            y = self.rec_file[0:nFrames,:,:]

            y_fill = y[0].copy().reshape(-1)
            for _ in range(self.nFrames-nFrames):
                # print(y.shape, y[0].shape)
                # copy top 1 to fill 
                y = np.concatenate([y_fill, y.reshape(-1)])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # last 3 frames
        elif label_frame_num > self.totalFrames-middle_num-1:
            nFrames = self.nFrames - (label_frame_num % (self.totalFrames-middle_num-1))
            start_frame = min(label_frame_num - middle_num, self.totalFrames-middle_num-1)
            y = self.rec_file[start_frame:start_frame+nFrames, :, :]
            
            y_fill = y[-1].copy().reshape(-1)
            for _ in range(self.nFrames-nFrames):
                y = np.concatenate([y.reshape(-1), y_fill])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # middle frames
        else:
            # number of frames to be read
            # nFrames = self.nFrames
            start_frame = label_frame_num - middle_num
            y = self.rec_file[start_frame:start_frame+self.nFrames, :, :]

            

        ### read label data
        label = self.label_file[label_frame_num, :, :]
        
        # print(label.size)
        # label = np.array(label)
        # print(type(label))

        ### cut
        cut_num = idx % ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        w, h = cut_num // (self.width//self.width_cut), cut_num % (self.width//self.width_cut)

        # print(idx)
        # print(y.shape, label.shape)

        input_ = y[middle_num, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)


        label_ = label[w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        # print(input_.shape, label_.shape)
        
        neighbor = y[[2,4,1,5,0,6], w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(6, -1, self.heigh_cut, self.width_cut)
        # neighbor = np.array(0)


        return torch.from_numpy(input_/255).float(), torch.from_numpy(neighbor/255).float(), torch.from_numpy(label_/255).float()
        # return torch.from_numpy(input_/255).float(), 0, torch.from_numpy(label_/255).float()

    def __len__(self):
        return self.len


class DataYUV_Fn(data.Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=6, width=1920, heigh=1080, width_cut=480, heigh_cut=270):
        super(DataYUV_Fn, self).__init__()
        self.nFrames = nFrames
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = heigh
        self.width_cut = width_cut
        self.heigh_cut = heigh_cut
        self.rec_file = rec_y
        self.label_file = label_y
        # self.rec_file = rec_y
        # self.label_file = label_y        
        self.len = (width//width_cut)*(heigh//heigh_cut)*totalFrames

    def get_referance_6(self, poc):
        if poc > 2 and poc < self.totalFrames-3:
            return [poc-3, poc-2, poc-1, poc+1, poc+2, poc+3]
        elif poc == 0:
            return [poc, poc, poc, poc+1, poc+2, poc+3]
        elif poc == 1:
            return [poc-1, poc, poc, poc+1, poc+2, poc+3]
        elif poc == 2:
            return [poc-2, poc-1, poc, poc+1, poc+2, poc+3]
        elif poc == self.totalFrames - 1:
            return [poc-3, poc-2, poc-1, poc, poc, poc]
        elif poc == self.totalFrames - 2:
            return [poc-3, poc-2, poc-1, poc, poc, poc+1]
        else: #poc == self.totalFrames - 3:
            return [poc-3, poc-2, poc-1, poc, poc+1, poc+2]

    def get_referance_4(self, poc):
        if poc > 1 and poc < self.totalFrames-2:
            return [poc-2, poc-1, poc+1, poc+2]
        elif poc == 0:
            return [ poc, poc, poc+1, poc+2]
        elif poc == 1:
            return [poc-1, poc, poc+1, poc+2]
        elif poc == self.totalFrames - 1:
            return [poc-2, poc-1, poc, poc]
        elif poc == self.totalFrames - 2:
            return [poc-2, poc-1, poc, poc+1]
        
    def get_referance_2(self, poc):
        if poc > 0 and poc < self.totalFrames-1:
            return [poc-1, poc+1]
        elif poc == 0:
            return [poc, poc+1]
        elif poc == self.totalFrames - 1:
            return [poc-1, poc]
        
    def get_referance_8(self, poc):
        if poc > 3 and poc < self.totalFrames-4:
            return [poc-4, poc-3, poc-2, poc-1, poc+1, poc+2, poc+3, poc+4]
        elif poc == 0:
            return [poc, poc, poc, poc, poc+1, poc+2, poc+3, poc+4]
        elif poc == 1:
            return [poc-1, poc, poc, poc, poc+1, poc+2, poc+3, poc+4]
        elif poc == 2:
            return [poc-2, poc-1, poc, poc, poc+1, poc+2, poc+3, poc+4]
        elif poc == 3:
            return [poc-3, poc-2, poc-1, poc, poc+1, poc+2, poc+3, poc+4]            
        elif poc == self.totalFrames - 1:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc, poc, poc]
        elif poc == self.totalFrames - 2:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc, poc, poc+1]
        elif poc == self.totalFrames - 3:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc, poc+1, poc+2]
        elif poc == self.totalFrames - 4:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc+1, poc+2, poc+3]

    def __getitem__(self, idx):
        label_idx = idx // ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))     # label_idx == picture of count
        ### cut
        cut_num = idx % ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        w, h = cut_num // (self.width//self.width_cut), cut_num % (self.width//self.width_cut)
        ### get label
        label = self.label_file[label_idx, :, :]
        label_ = label[w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        ### get referance
        if self.nFrames == 6:
            ref_idx = self.get_referance_6(label_idx)
        elif self.nFrames == 4:
            ref_idx = self.get_referance_4(label_idx)
        elif self.nFrames == 2:
            ref_idx = self.get_referance_2(label_idx)
        elif self.nFrames == 8:
            ref_idx = self.get_referance_8(label_idx)
            
        referance = self.rec_file[ref_idx, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, 1, self.heigh_cut, self.width_cut)
        ### get input
        input_ = self.rec_file[label_idx, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)

        return torch.from_numpy(input_/255).float(), torch.from_numpy(referance/255).float(), torch.from_numpy(label_/255).float()
        # return (input_/255).float(), (referance/255).float(), (label_/255).float()
        # return input_, referance, label_


    def __len__(self):
        return self.len


class DataYUV_F0(data.Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=6, width=1920, heigh=1080, width_cut=480, heigh_cut=270):
        super(DataYUV_F0, self).__init__()
        self.nFrames = nFrames
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = heigh
        self.width_cut = width_cut
        self.heigh_cut = heigh_cut
        self.rec_file = rec_y
        self.label_file = label_y
        # self.rec_file = rec_y
        # self.label_file = label_y        
        self.len = (width//width_cut)*(heigh//heigh_cut)*totalFrames

    def get_referance_6(self, poc):
        if poc > 2 and poc < self.totalFrames-3:
            return [poc-3, poc-2, poc-1, poc+1, poc+2, poc+3]
        elif poc == 0:
            return [poc, poc, poc, poc+1, poc+2, poc+3]
        elif poc == 1:
            return [poc-1, poc, poc, poc+1, poc+2, poc+3]
        elif poc == 2:
            return [poc-2, poc-1, poc, poc+1, poc+2, poc+3]
        elif poc == self.totalFrames - 1:
            return [poc-3, poc-2, poc-1, poc, poc, poc]
        elif poc == self.totalFrames - 2:
            return [poc-3, poc-2, poc-1, poc, poc, poc+1]
        else: #poc == self.totalFrames - 3:
            return [poc-3, poc-2, poc-1, poc, poc+1, poc+2]

    def get_referance_4(self, poc):
        if poc > 1 and poc < self.totalFrames-2:
            return [poc-2, poc-1, poc+1, poc+2]
        elif poc == 0:
            return [ poc, poc, poc+1, poc+2]
        elif poc == 1:
            return [poc-1, poc, poc+1, poc+2]
        elif poc == self.totalFrames - 1:
            return [poc-2, poc-1, poc, poc]
        elif poc == self.totalFrames - 2:
            return [poc-2, poc-1, poc, poc+1]
        
    def get_referance_2(self, poc):
        return [poc, poc]

    def get_referance_8(self, poc):
        if poc > 3 and poc < self.totalFrames-4:
            return [poc-4, poc-3, poc-2, poc-1, poc+1, poc+2, poc+3, poc+4]
        elif poc == 0:
            return [poc, poc, poc, poc, poc+1, poc+2, poc+3, poc+4]
        elif poc == 1:
            return [poc-1, poc, poc, poc, poc+1, poc+2, poc+3, poc+4]
        elif poc == 2:
            return [poc-2, poc-1, poc, poc, poc+1, poc+2, poc+3, poc+4]
        elif poc == 3:
            return [poc-3, poc-2, poc-1, poc, poc+1, poc+2, poc+3, poc+4]            
        elif poc == self.totalFrames - 1:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc, poc, poc]
        elif poc == self.totalFrames - 2:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc, poc, poc+1]
        elif poc == self.totalFrames - 3:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc, poc+1, poc+2]
        elif poc == self.totalFrames - 4:
            return [poc-4, poc-3, poc-2, poc-1, poc, poc+1, poc+2, poc+3]

    def __getitem__(self, idx):
        label_idx = idx // ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))     # label_idx == picture of count
        ### cut
        cut_num = idx % ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        w, h = cut_num // (self.width//self.width_cut), cut_num % (self.width//self.width_cut)
        ### get label
        label = self.label_file[label_idx, :, :]
        label_ = label[w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        ### get referance
        if self.nFrames == 6:
            ref_idx = self.get_referance_6(label_idx)
        elif self.nFrames == 4:
            ref_idx = self.get_referance_4(label_idx)
        elif self.nFrames == 2:
            ref_idx = self.get_referance_2(label_idx)
        elif self.nFrames == 8:
            ref_idx = self.get_referance_8(label_idx)
            
        referance = self.rec_file[ref_idx, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, 1, self.heigh_cut, self.width_cut)
        ### get input
        input_ = self.rec_file[label_idx, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)

        return torch.from_numpy(input_/255).float(), torch.from_numpy(referance/255).float(), torch.from_numpy(label_/255).float()
        # return (input_/255).float(), (referance/255).float(), (label_/255).float()
        # return input_, referance, label_


    def __len__(self):
        return self.len


class DataYUV_gpu(data.Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=7, width=1920, heigh=1080, width_cut=480, heigh_cut=270):
        super(DataYUV_gpu, self).__init__()
        self.nFrames = nFrames
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = heigh
        self.width_cut = width_cut
        self.heigh_cut = heigh_cut
        self.rec_file = torch.from_numpy(rec_y/255).float().cuda()
        self.label_file = torch.from_numpy(label_y/255).float().cuda()
        # self.rec_file = rec_y
        # self.label_file = label_y        
        self.len = (width//width_cut)*(heigh//heigh_cut)*totalFrames

    def get_referance(self, poc):
        if poc > 2 and poc < self.totalFrames-3:
            return [poc-3, poc-2, poc-1, poc+1, poc+2, poc+3]
        elif poc == 0:
            return [poc, poc, poc, poc+1, poc+2, poc+3]
        elif poc == 1:
            return [poc-1, poc, poc, poc+1, poc+2, poc+3]
        elif poc == 2:
            return [poc-2, poc-1, poc, poc+1, poc+2, poc+3]
        elif poc == self.totalFrames - 1:
            return [poc-3, poc-2, poc-1, poc, poc, poc]
        elif poc == self.totalFrames - 2:
            return [poc-3, poc-2, poc-1, poc, poc, poc+1]
        else: #poc == self.totalFrames - 3:
            return [poc-3, poc-2, poc-1, poc, poc+1, poc+2]

    def __getitem__(self, idx):
        label_idx = idx // ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))     # label_idx == picture of count
        ### cut
        cut_num = idx % ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        w, h = cut_num // (self.width//self.width_cut), cut_num % (self.width//self.width_cut)
        ### get label
        label = self.label_file[label_idx, :, :]
        label_ = label[w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        ### get referance
        ref_idx = self.get_referance(label_idx)

        referance = self.rec_file[ref_idx, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, 1, self.heigh_cut, self.width_cut)
        ### get input
        input_ = self.rec_file[label_idx, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)

        # return torch.from_numpy(input_/255).float(), torch.from_numpy(referance/255).float(), torch.from_numpy(label_/255).float()
        # return (input_/255).float(), (referance/255).float(), (label_/255).float()
        return input_, referance, label_


    def __len__(self):
        return self.len


class DataYUV_v2(data.Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=1, width=1920, heigh=1080, width_cut=480, heigh_cut=270):
        super(DataYUV_v2, self).__init__()
        self.nFrames = nFrames
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = heigh
        self.width_cut = width_cut
        self.heigh_cut = heigh_cut
        self.rec_file = rec_y
        self.label_file = label_y
        self.len = (width//width_cut)*(heigh//heigh_cut)*totalFrames


    def __getitem__(self, idx):
        label_frame_num = idx // ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        middle_num = self.nFrames//2   # 3

        ### read multi input data
        # top 3 frames
        if label_frame_num < middle_num:
            # number of frames to be read
            nFrames = middle_num + 1 + (label_frame_num % middle_num)
            y = self.rec_file[0:nFrames,:,:]

            y_fill = y[0].copy().reshape(-1)
            for _ in range(self.nFrames-nFrames):
                # print(y.shape, y[0].shape)
                # copy top 1 to fill 
                y = np.concatenate([y_fill, y.reshape(-1)])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # last 3 frames
        elif label_frame_num > self.totalFrames-middle_num-1:
            nFrames = self.nFrames - (label_frame_num % (self.totalFrames-middle_num-1))
            start_frame = min(label_frame_num - middle_num, self.totalFrames-middle_num-1)
            y = self.rec_file[start_frame:start_frame+nFrames, :, :]
            
            y_fill = y[-1].copy().reshape(-1)
            for _ in range(self.nFrames-nFrames):
                y = np.concatenate([y.reshape(-1), y_fill])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # middle frames
        else:
            # number of frames to be read
            # nFrames = self.nFrames
            start_frame = label_frame_num - middle_num
            y = self.rec_file[start_frame:start_frame+self.nFrames, :, :]

            

        ### read label data
        label = self.label_file[label_frame_num, :, :]
        
        # print(label.size)
        # label = np.array(label)
        # print(type(label))

        ### cut
        cut_num = idx % ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        w, h = cut_num // (self.width//self.width_cut), cut_num % (self.width//self.width_cut)
        # print(idx, w, h)
        # print(y.shape, label.shape)

        input_ = y[middle_num, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)


        label_ = label[w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        # print(input_.shape, label_.shape)
        
        # neighbor = y[[2,4,1,5,0,6], h*self.heigh_cut:(h+1)*self.heigh_cut, w*self.width_cut:(w+1)*self.width_cut].reshape(6, -1, self.heigh_cut, self.width_cut)
        # neighbor = np.array(0)


        # return torch.from_numpy(input_/255).float(), torch.from_numpy(neighbor/255).float(), torch.from_numpy(label_/255).float()
        return torch.from_numpy(input_/255).float(), 0, torch.from_numpy(label_/255).float()

    def __len__(self):
        return self.len


class DataYUV_v3(data.Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=7, width=1920, heigh=1080, width_cut=1920, heigh_cut=1080):
        super(DataYUV_v3, self).__init__()
        self.nFrames = nFrames
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = heigh
        self.width_cut = width_cut
        self.heigh_cut = heigh_cut
        self.rec_file = rec_y
        self.label_file = label_y
        self.len = (width//width_cut)*(heigh//heigh_cut)*totalFrames


    def __getitem__(self, idx):
        frame_idx = idx // ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))

        # top 6 frames
        if frame_idx < self.nFrames-1:
            y = self.rec_file[0:frame_idx+1,:,:]
            y_fill = y[frame_idx].copy().reshape(-1)
            for _ in range(self.nFrames-frame_idx-1):
                # print(y.shape, y[0].shape)
                # copy top 1 to fill 
                y = np.concatenate([y.reshape(-1), y_fill])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
            # print(y.shape)
        
        else:
            # number of frames to be read
            # nFrames = self.nFrames
            y = self.rec_file[frame_idx-self.nFrames+1:frame_idx+1, :, :]
            # print(y.shape)
          
        ### read label data
        label = self.label_file[frame_idx, :, :]

        ### cut
        cut_num = idx % ((self.width//self.width_cut)*(self.heigh//self.heigh_cut))
        w, h = cut_num // (self.width//self.width_cut), cut_num % (self.width//self.width_cut)

        # print(idx)
        # print(y.shape, label.shape)

        input_ = y[-1, w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        label_ = label[w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        # print(input_.shape, label_.shape)
        neighbor = y[[5,4,3,2,1,0], w*self.heigh_cut:(w+1)*self.heigh_cut, h*self.width_cut:(h+1)*self.width_cut].reshape(6, -1, self.heigh_cut, self.width_cut)
        # neighbor = np.array(0)

        # print(input_.shape, label_.shape, neighbor.shape)
        return torch.from_numpy(input_/255).float(), torch.from_numpy(neighbor/255).float(), torch.from_numpy(label_/255).float()
        # return torch.from_numpy(input_/255).float(), 0, torch.from_numpy(label_/255).float()

    def __len__(self):
        return self.len


# data for pre-train
class Data_codec_isolate(data.Dataset):
    def __init__(self, qp, yaml_obj1, yaml_obj2, yaml_obj3, train=True,  transform=transforms.ToTensor()):
        self.width = 144
        # self.ref_num = ref_num
        self.qp = qp
        self.transform = transform
        # self.yaml_obj = yaml_obj
        self.yaml_obj1 = yaml_obj1
        self.yaml_obj2 = yaml_obj2
        self.yaml_obj3 = yaml_obj3
        self.len = 57307
        # self.len = 36066
        # self.len = 15726

        self.GT_env = None
        self.Rec_env = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if idx < 15727:
            name_GT = '/home/disk1/lilei/H265-overfit/data/derf/part1_GT'
            name_high = '/home/disk1/lilei/H265-overfit/data/derf/part1_ra_QP{}'.format(self.qp)
            # name_low = '/data/disk2/Datasets/Vimeo/part1/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp-4)
            key_idx = self.yaml_obj1[idx]
            # rec_idx = [self.yaml_obj1[i] for i in range(idx-3, idx+4)]
        elif idx > 15726 and idx < 36067 :
            name_GT = '/home/disk1/lilei/H265-overfit/data/derf/part2_GT'
            name_high = '/home/disk1/lilei/H265-overfit/data/derf/part2_ra_QP{}'.format(self.qp)
            # name_low = '/data/disk2/Datasets/Vimeo/part2/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp-4)
            key_idx = self.yaml_obj2[idx-15727]
            # rec_idx = [self.yaml_obj2[i] for i in range(idx-15727-1, idx-15727+4)]
        else :
            name_GT = '/home/disk1/lilei/H265-overfit/data/derf/part3_GT'
            name_high = '/home/disk1/lilei/H265-overfit/data/derf/part3_ra_QP{}'.format(self.qp)
            # name_low = '/data/disk2/Datasets/Vimeo/part3/Vimeo_packed/lmdb/ra_QP{}'.format(self.qp-4)
            key_idx = self.yaml_obj3[idx-36067]

        GT_env = lmdb.open(name_GT, readonly=True, lock=False, readahead=False,
                                    meminit=False)
        High_env = lmdb.open(name_high, readonly=True, lock=False, readahead=False,
                                    meminit=False)
        #GT_nvideo_nframe
        n_frame = random.randint(0,16)
        GT_key = 'GT_'+key_idx+'_'+ str(n_frame).rjust(3, '0')
        # Rec_key_high = 'Rec_'+key_idx+'_'+ str(n_frame).rjust(3, '0')
        Rec_key_high = []
        for i in range(n_frame-3, n_frame+4):
            if i >=0:
                if i <= 16:
                    Rec_key_high.append('Rec_'+key_idx+'_'+ str(i).rjust(3, '0'))
                else:
                    Rec_key_high.append('Rec_'+key_idx+'_'+ str(16).rjust(3, '0'))
            else:
                Rec_key_high.append('Rec_'+key_idx+'_'+ str(0).rjust(3, '0'))

        # nh = 60
        cut_width = 60
        
        #### get the GT image (as the center frame)
        # print(GT_key)
        img_GT = util._read_img_lmdb(GT_env, GT_key, (1, int(self.width), int(self.width)))

        img_Rec_high = []
        for name in Rec_key_high:
            # print(name)
            img_Rec_high.append(util._read_img_lmdb(High_env, name, (1, int(self.width), int(self.width))))
        img_Rec_high = np.concatenate(img_Rec_high, 2)

        ### randomly crop
        rnd_w = random.randint(0, max(0, int(self.width) - cut_width))
        rnd_h = random.randint(0, max(0, int(self.width) - cut_width))
        img_GT = img_GT[rnd_h:rnd_h + cut_width, rnd_w:rnd_w + cut_width, :]
        img_Rec_high = img_Rec_high[rnd_h:rnd_h + cut_width, rnd_w:rnd_w + cut_width,:]
        # neighbor_ = img_Rec_high[:,:,[0,1,2,4,5,6]]    // change neighbor
        neighbor_ = img_Rec_high[:,:,[2,4]]
        input_ = img_Rec_high[:,:,[3]]
        # img_Rec_high = util._read_img_lmdb(High_env, Rec_key_high, (1, int(self.width), int(self.width)))

        High_env.close()
        GT_env.close()

        # [HWC] to [CHW]   [NHWC] to [NCHW]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT/255.0, (2, 0, 1)))).float()
        input_ = torch.from_numpy(np.ascontiguousarray(np.transpose(input_/255.0, (2, 0, 1)))).float()
        neighbor_ = torch.from_numpy(np.ascontiguousarray(np.transpose(neighbor_/255.0, (2, 0, 1)).reshape(2,1,60,60))).float()
        # img_Rec_high = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Rec_high/255.0, (2, 0, 1)))).float()

        return  input_, neighbor_, img_GT






if __name__ == "__main__":
    width_cut=80
    heigh_cut=135
    width, heigh = 1920, 1080
    rec_path='../data/videos/BasketballDrive_1920x1080_50_000to049_QP22_IP_rec.yuv'
    label_path='../data/videos/BasketballDrive_1920x1080_50_000to049.yuv'
    rec_file = open(rec_path, 'rb')
    label_file = open(label_path, 'rb')
    rec_y, _, _ = YUVread(rec_file, [heigh, width], frame_num=50, start_frame=0)
    label_y, _, _ = YUVread(label_file, [heigh, width], frame_num=50, start_frame=0)

    dataset = DataYUV_v2(rec_y=rec_y, label_y=label_y, width_cut=width_cut, heigh_cut=heigh_cut)
    all_y, all_label = [], []
    all_neighbor_0, all_neighbor_1, all_neighbor_2, all_neighbor_3, all_neighbor_4, all_neighbor_5 = [], [], [], [], [], []

    for i in range(100):
        
        y, neighbor, label = dataset.__getitem__(i)
        # print(y.dtype, neighbor.dtype, label.dtype)
        all_y.append(y)
        # all_neighbor_0.append(neighbor[0]) #.reshape(1,270,480))
        # all_neighbor_1.append(neighbor[1]) #.reshape(1,270,480))
        # all_neighbor_2.append(neighbor[2]) #.reshape(1,270,480))
        # all_neighbor_3.append(neighbor[3]) #.reshape(1,270,480))
        # all_neighbor_4.append(neighbor[4]) #.reshape(1,270,480))
        # all_neighbor_5.append(neighbor[5]) #.reshape(1,270,480))
        all_label.append(label)


    torchvision.utils.save_image(all_y, 'result/input.png', nrow=3)
    # torchvision.utils.save_image(all_neighbor_0, 'result/all_neighbor_0.png', nrow=3)
    # torchvision.utils.save_image(all_neighbor_1, 'result/all_neighbor_1.png', nrow=3)
    # torchvision.utils.save_image(all_neighbor_2, 'result/all_neighbor_2.png', nrow=3)
    # torchvision.utils.save_image(all_neighbor_3, 'result/all_neighbor_3.png', nrow=3)
    # torchvision.utils.save_image(all_neighbor_4, 'result/all_neighbor_4.png', nrow=3)
    # torchvision.utils.save_image(all_neighbor_5, 'result/all_neighbor_5.png', nrow=3)
    torchvision.utils.save_image(all_label, 'result/label.png', nrow=3)


    #save_grid_img(make_grid(input_lr), 'input_lr.png')
    #save_grid_img(make_grid(input_lr), 'output_lr.png')
    #save_grid_img(make_grid(input_lr), 'output_sr.png')
    # cv2.waitKey(0)
    #    a, b, c = dataset.__getitem__(20000)
    # print(input_lr.shape, output_lr.shape, output_sr.shape)
    print(len(dataset))
    
