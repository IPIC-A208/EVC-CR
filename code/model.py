import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import yaml
import time
import os







class Model():
    def __init__(self, device=None, lr=0.001, train=True, model=None, optim='Adam'):
        self.device = device if device is not None else torch.device('cuda')
        self.net = model.to(self.device)
        self.net.train() if train else self.net.eval()
        self.gain_max = -30.0
        self.gain = -20.0
        # self.psnr
        if optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.lossF = torch.nn.L1Loss().to(self.device)
        # self.loss = torch
        # self.set_random_seed()
        # self.weight_init()
        self.t0 = time.time()
        self.epoch = 0
        self.lr = lr
        

    def feed_data(self, x, neighbor, label):
        self.x = x.to(self.device)
        self.neighbor = neighbor.to(self.device)
        self.label = label.to(self.device)

    def print_network(self):
        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()
        print(self.net)
        print('Total number of parameters: %d' % num_params)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.out = self.net(self.x, self.neighbor)
        # self.out = self.net(self.x)
        self.loss = self.lossF(self.out, self.label)
        self.loss.backward()
        self.optimizer.step()
    
    def log(self, step, epoch, length, args, writer=None):
        psnr_pre = -20 * ((self.x - self.label).pow(2).mean().pow(0.5)).log10()
        psnr_aft = -20 * ((self.out - self.label).pow(2).mean().pow(0.5)).log10()
        psnr_gain = psnr_aft - psnr_pre
        if writer is not None:
            writer.add_scalar("psnr_pre", psnr_pre.item(), length*(epoch-1)+step)
            writer.add_scalar("psnr_aft", psnr_aft.item(), length*(epoch-1)+step)
            writer.add_scalar("loss", self.loss.item(), length*(epoch-1)+step)
        if epoch > self.epoch:
            self.gain = self.gain/length
            self.epoch = epoch
            print('epoch[{}] : psnr_gain: {:.6f} | psnr_gain_max: {:.6f}'.format(epoch-1, self.gain, self.gain_max))

            if self.gain > (self.gain_max + 0.001):
                name = '{}_{}_allFrame_{}_QP{}'.format(args.model, args.sequence, args.frame, args.qp)
                self._log_maxgain(psnr_gain, epoch, name)
                self.gain_max = self.gain
                self.save_network(name)

            self.gain = psnr_gain.item()
        else:
            self.gain += psnr_gain.item()
        print("Epoch[{}][{}/{}] | lr:{:.6f} | loss: {:.4f} | psnr_aft: {:.4f} psnr_pre: {:.4f} psnr_gain: {:.4f}".format(epoch, step+1, length, self.optimizer.param_groups[0]['lr'], self.loss.item(), psnr_aft.item(), psnr_pre.item(), psnr_gain.item()))


    def _log_maxgain(self, psnr_gain, epoch, name):
        if not os.path.exists('log'):
            os.makedirs('log')
        name = os.path.join('log', name+'.yaml')
        f = open(name, 'a', encoding='utf-8')
        message = 'epoch_{}_lr_{:.6}_hour_{:.6f}_psnrgin : {:.6f}'.format(epoch, self.lr, (time.time()-self.t0)/3600, self.gain_max)
        yaml.dump(message, f)
        f.close()

    def weight_init(self):
        classname = self.net.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('ConvTranspose') != -1):
            torch.nn.init.normal_(self.net.weight.data, 0.0, 1.0)        

    def save_network(self, name):
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        name = os.path.join('checkpoint', '{}_gain_{:.4f}_.pth'.format(name, self.gain_max))
        torch.save(self.net.state_dict(), name)

    def resume_network(self, name):
        name = os.path.join('checkpoint', name)
        parameters = torch.load(name)
        self.net.load_state_dict(parameters)

    def test(self):
        self.net.eval()
        self.out = self.net(self.x, self.neighbor)
        self.loss = self.lossF(self.out, self.label)
        return self.out

    def adajust_lr(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr

    def set_random_seed(self, seed=1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



