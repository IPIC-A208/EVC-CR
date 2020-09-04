import torch.utils.data as data
import torch


class MDataYUV(data.Dataset):
    def __init__(self, rec_y, label_y, totalFrames=50, nFrames=6, width=1920, heigh=1080, width_cut=480, heigh_cut=270):
        super(MDataYUV, self).__init__()
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



