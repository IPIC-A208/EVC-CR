from yuv_io import *
from cut_frame import *

from torch.utils.data.dataset import Dataset
import torch
from torch.utils.data import DataLoader


class FrameDataset(Dataset):
    def __init__(self, im_label, im_input):
        super(FrameDataset, self).__init__()
        self.label_data = im_label
        self.input_data = im_input

    def __getitem__(self, index):
        label = self.label_data[index]
        input = self.input_data[index]
        return label, input

    def __len__(self):
        return self.label_data.size(0)


class MultiFrameDataset(Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=7, width=1920, height=1080, width_cut=480, height_cut=270):
        super(MultiFrameDataset, self).__init__()
        self.nFrames = nFrames + 1
        self.totalFrames = totalFrames
        self.width = width
        self.heigh = height
        self.width_cut = width_cut
        self.heigh_cut = height_cut
        self.rec_file = rec_y
        self.label_file = label_y
        self.len = (width // width_cut) * (height // height_cut) * totalFrames

    def __getitem__(self, idx):
        # label_frame_num = idx//(self.width*self.heigh//self.width_cut//self.heigh_cut)
        label_frame_num = idx // ((self.width // self.width_cut) * (self.heigh // self.heigh_cut))
        middle_num = self.nFrames // 2  # 3
        neighbor_frame = self.nFrames - 1
        ### read multi input data
        # top 3 frames
        if label_frame_num < middle_num:
            # number of frames to be read
            nFrames = middle_num + 1 + (label_frame_num % middle_num)
            y = self.rec_file[0:nFrames, :, :]

            y_fill = y[0].copy().reshape(-1)
            for _ in range(self.nFrames - nFrames):
                # print(y.shape, y[0].shape)
                # copy top 1 to fill
                y = np.concatenate([y_fill, y.reshape(-1)])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # last 3 frames
        elif label_frame_num > self.totalFrames - middle_num - 1:
            nFrames = self.nFrames - (label_frame_num % (self.totalFrames - middle_num - 1))
            start_frame = min(label_frame_num - middle_num, self.totalFrames - middle_num - 1)
            y = self.rec_file[start_frame:start_frame + nFrames, :, :]

            y_fill = y[-1].copy().reshape(-1)
            for _ in range(self.nFrames - nFrames):
                y = np.concatenate([y.reshape(-1), y_fill])
            y = np.reshape(y, [self.nFrames, self.heigh, self.width])
        # middle frames
        else:
            # number of frames to be read
            # nFrames = self.nFrames
            start_frame = label_frame_num - middle_num
            y = self.rec_file[start_frame:start_frame + self.nFrames, :, :]

        ### read label data
        label = self.label_file[label_frame_num, :, :]

        # print(label.size)
        # label = np.array(label)
        # print(type(label))

        ### cut
        cut_num = idx % ((self.width // self.width_cut) * (self.heigh // self.heigh_cut))
        w, h = cut_num // (self.width // self.width_cut), cut_num % (self.width // self.width_cut)

        # print(idx)
        # print(y.shape, label.shape)

        input_ = y[middle_num, w * self.heigh_cut:(w + 1) * self.heigh_cut,
                 h * self.width_cut:(h + 1) * self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)

        label_ = label[w * self.heigh_cut:(w + 1) * self.heigh_cut,
                 h * self.width_cut:(h + 1) * self.width_cut].reshape(-1, self.heigh_cut, self.width_cut)
        # print(input_.shape, label_.shape)
        if  neighbor_frame == 2:
            neighbor = y[[0,2], w * self.heigh_cut:(w + 1) * self.heigh_cut,
                    h * self.width_cut:(h + 1) * self.width_cut].reshape(2, -1, self.heigh_cut, self.width_cut)
        else:
            neighbor = y[[2,4,1,5,0,6], w * self.heigh_cut:(w + 1) * self.heigh_cut,
                    h * self.width_cut:(h + 1) * self.width_cut].reshape(6, -1, self.heigh_cut, self.width_cut)
        
    
        return torch.from_numpy(input_ / 255).float(), torch.from_numpy(neighbor / 255).float(), torch.from_numpy(
            label_ / 255).float()
        # return torch.from_numpy(input_/255).float(), 0, torch.from_numpy(label_/255).float()

    def __len__(self):
        return self.len






if __name__ == '__main__':
    # input_path = './data/videos/BasketballDrive_1920x1080_50_000to049_QP37_IP_rec_H266.yuv'
    # label_path = './data/videos/BasketballDrive_1920x1080_50_000to049.yuv'
    input_path = './data/videos/BasketballDrive_1920x1080_50_000to049_QP37_IP_rec_H266.yuv'
    label_path = './data/videos/BasketballDrive_1920x1080_50_000to049.yuv'

    height = 1080
    width = 1920

    initial_epoch = 0
    n_epoch = 1
    batch_size = 1

    loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')

    # for epoch in range(initial_epoch, n_epoch):
    #     input_data, label_data, count = cut_frame(input_path, label_path, [height, width], 41, 36)
    #
    #     input_data = torch.from_numpy(input_data.transpose((0, 3, 1, 2)) / 255)
    #     label_data = torch.from_numpy(label_data.transpose((0, 3, 1, 2)) / 255)
    #
    #     DDataset = FrameDataset(label_data, input_data)
    #     DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
    #
    #     for n_count, batch_yx in enumerate(DLoader):
    #         batch_x, batch_y = batch_yx[1], batch_yx[0]
    #         loss = loss_fn(batch_x, batch_y)
    #         psnr = np.multiply(10.0, np.log(1.0 * 1.0 / loss) / np.log(10.0))
    #         print('Loss : %4f    PSNR : %4f' % (loss, psnr))

    im_input, _, _ = YUVread(input_path, [height, width])
    im_label, _, _ = YUVread(label_path, [height, width])
    input_data = np.expand_dims(im_input, axis=3)
    label_data = np.expand_dims(im_label, axis=3)
    input_data, label_data = input_data.astype('float32') / 255.0, label_data.astype('float32') / 255.0
    input_data = torch.from_numpy(input_data.transpose((0, 3, 1, 2)))
    label_data = torch.from_numpy(label_data.transpose((0, 3, 1, 2)))

    EvalDataset = FrameDataset(im_label=label_data, im_input=input_data)
    EvalLoader = DataLoader(dataset=EvalDataset, num_workers=4, drop_last=False, batch_size=1, shuffle=False)

    for n_count, batch_yx in enumerate(EvalLoader):
        batch_x, batch_y = batch_yx[1], batch_yx[0]
        loss = loss_fn(batch_x, batch_y)
        psnr = np.multiply(10.0, np.log(1.0 * 1.0 / loss) / np.log(10.0))
        print('Frame%d/%d   Loss : %4f    PSNR : %4f' % ((n_count+1), len(input_data), loss, psnr))

